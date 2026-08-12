"""실험 1: 미니 CLIP류 이미지<->텍스트 어댑터.

Module A = frozen 이미지 인코더 (ResNet18, 나중에 ResNet34로 교체 = "업데이트")
Module B = frozen 텍스트 인코더 (sentence-transformers, 고정)
Interface = 학습 가능한 Adapter (이미지 임베딩 -> 텍스트 임베딩 공간으로 projection)

Task: CIFAR-10 이미지를 "a photo of a {class}" 프롬프트 10개 중 가장 가까운 것으로
분류 (CLIP식 zero-shot 방식). Quality = top-1 accuracy.

흐름:
  1) ResNet18로 이미지 임베딩 추출 -> adapter 처음부터 학습 (cold start curve)
  2) 이미지 인코더를 ResNet34로 교체(=module A 업데이트) -> 같은 adapter로 즉시 평가
     (quality가 얼마나 떨어지는지 측정 = degraded_quality)
  3) adapter만 재학습 -> quality 회복 곡선 측정 (retrain_time)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models

sys.path.insert(0, str(Path(__file__).parent))
from common import Adapter, RetrainHarness, StepFn, device

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

N_TRAIN = 4000
N_TEST = 1000
BATCH_SIZE = 256
TEMPERATURE = 0.07


CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def load_cifar() -> tuple[Dataset, Dataset]:
    tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=tfm)
    test = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=tfm)
    train = torch.utils.data.Subset(train, range(N_TRAIN))
    test = torch.utils.data.Subset(test, range(N_TEST))
    return train, test


@torch.no_grad()
def extract_embeddings(encoder: nn.Module, dataset: Dataset, dev: torch.device,
                        batch_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    encoder.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    embeds, labels = [], []
    for imgs, ys in loader:
        imgs = imgs.to(dev)
        feats = encoder(imgs).flatten(1)
        embeds.append(feats.cpu())
        labels.append(ys)
    return torch.cat(embeds), torch.cat(labels)


def make_image_encoder(arch: str, dev: torch.device) -> nn.Module:
    if arch == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif arch == "resnet34":
        net = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(arch)
    net.fc = nn.Identity()
    net.to(dev)
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def encode_text_prompts(dev: torch.device) -> torch.Tensor:
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("all-MiniLM-L6-v2", device=str(dev))
    prompts = [f"a photo of a {c}" for c in CIFAR_CLASSES]
    with torch.no_grad():
        embeds = st.encode(prompts, convert_to_tensor=True, device=str(dev))
    return F.normalize(embeds, dim=-1).cpu()


def accuracy(adapter: nn.Module, img_embeds: torch.Tensor, labels: torch.Tensor,
             text_embeds: torch.Tensor, dev: torch.device) -> float:
    adapter.eval()
    with torch.no_grad():
        proj = F.normalize(adapter(img_embeds.to(dev)), dim=-1)
        logits = proj @ text_embeds.to(dev).T
        preds = logits.argmax(dim=-1).cpu()
    adapter.train()
    return (preds == labels).float().mean().item()


def make_step_fn(adapter: nn.Module, optimizer: torch.optim.Optimizer,
                  train_embeds: torch.Tensor, train_labels: torch.Tensor,
                  text_embeds: torch.Tensor, dev: torch.device,
                  eval_embeds: torch.Tensor, eval_labels: torch.Tensor) -> StepFn:
    n = train_embeds.size(0)

    def step_fn(step):
        idx = torch.randint(0, n, (BATCH_SIZE,))
        img_batch = train_embeds[idx].to(dev)
        label_batch = train_labels[idx].to(dev)
        proj = F.normalize(adapter(img_batch), dim=-1)
        logits = proj @ text_embeds.to(dev).T / TEMPERATURE
        loss = F.cross_entropy(logits, label_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        quality = accuracy(adapter, eval_embeds, eval_labels, text_embeds, dev)
        return loss.item(), quality

    return step_fn


def main() -> RetrainHarness:
    dev = device()
    print(f"[image_text] device={dev}")

    train_ds, test_ds = load_cifar()

    print("[image_text] extracting text prompt embeddings...")
    text_embeds = encode_text_prompts(dev)

    print("[image_text] loading ResNet18 (module A v1) and extracting image embeddings...")
    encoder_v1 = make_image_encoder("resnet18", dev)
    train_embeds_v1, train_labels = extract_embeddings(encoder_v1, train_ds, dev)
    test_embeds_v1, test_labels = extract_embeddings(encoder_v1, test_ds, dev)
    del encoder_v1
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    img_dim = train_embeds_v1.size(-1)
    txt_dim = text_embeds.size(-1)
    adapter = Adapter(img_dim, txt_dim).to(dev)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)

    harness = RetrainHarness("image_text")

    print("[image_text] phase 1: initial adapter training (ResNet18)...")
    step_fn = make_step_fn(adapter, optimizer, train_embeds_v1, train_labels, text_embeds, dev,
                            test_embeds_v1, test_labels)
    rec1 = harness.run("initial_train", step_fn, max_steps=300, eval_every=5,
                        recover_target=None, patience_stop=None)
    quality_before_swap = rec1.history[-1]["quality"]
    rec1.quality_before_swap = quality_before_swap
    print(f"  -> reached quality={quality_before_swap:.3f}")

    print("[image_text] swapping module A: ResNet18 -> ResNet34 (\"module update\")...")
    encoder_v2 = make_image_encoder("resnet34", dev)
    train_embeds_v2, _ = extract_embeddings(encoder_v2, train_ds, dev)
    test_embeds_v2, _ = extract_embeddings(encoder_v2, test_ds, dev)
    del encoder_v2
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    quality_after_swap = accuracy(adapter, test_embeds_v2, test_labels, text_embeds, dev)
    print(f"  -> quality immediately after swap (adapter NOT retrained) = {quality_after_swap:.3f}")

    print("[image_text] phase 2: retraining adapter against ResNet34 features...")
    step_fn2 = make_step_fn(adapter, optimizer, train_embeds_v2, train_labels, text_embeds, dev,
                             test_embeds_v2, test_labels)
    rec2 = harness.run("post_swap_retrain", step_fn2, max_steps=300, eval_every=5,
                        recover_target=0.95 * quality_before_swap, patience_stop=20)
    rec2.quality_before_swap = quality_before_swap
    rec2.quality_immediately_after_swap = quality_after_swap

    path = harness.save(RESULTS_DIR)
    print(f"[image_text] saved -> {path}")
    print(f"[image_text] retrain_time_sec={rec2.retrain_time_sec}, steps_to_recover={rec2.steps_to_recover}")
    return harness


if __name__ == "__main__":
    main()
