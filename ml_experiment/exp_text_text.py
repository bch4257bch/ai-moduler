"""실험 2: 두 문장 임베딩 모델(서로 다른 표현 공간) 사이의 텍스트 전용 어댑터.

Module A = frozen 문장 임베딩 모델 (all-MiniLM-L6-v2, 고정 = "검색엔진의 쿼리 인코더")
Module B = frozen 문장 임베딩 모델 (처음엔 paraphrase-MiniLM-L6-v2,
           나중에 all-MiniLM-L12-v2로 교체 = "업데이트된 검색 인덱스 인코더")
Interface = 학습 가능한 Adapter (A의 임베딩 공간 -> B의 임베딩 공간으로 translation)

Task: adapter(A(sentence))가 B(sentence)와 같은 문장을 가리키도록 in-batch
contrastive 학습 (label 불필요, self-supervised). Quality = batch 내 top-1
retrieval accuracy (자기 자신의 B 임베딩을 negative들 사이에서 찾아내는 비율).

일반 사용자가 흔히 쓰는 "문장 임베딩 -> 검색/분류" 파이프라인에서, 검색 인덱스
쪽 인코더가 새 버전으로 바뀌면 기존 학습된 인터페이스가 얼마나 깨지고 얼마나
빨리 재학습으로 회복되는지를 측정한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from common import Adapter, RetrainHarness, StepFn, device

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"

N_TRAIN = 3000
N_TEST = 800
BATCH_SIZE = 128
TEMPERATURE = 0.07


def load_sentences() -> tuple[list[str], list[str]]:
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/ag_news", split="train")
    texts = ds["text"][: N_TRAIN + N_TEST]
    return texts[:N_TRAIN], texts[N_TRAIN:N_TRAIN + N_TEST]


@torch.no_grad()
def encode_all(model_name: str, sentences: list[str], dev: torch.device,
                batch_size: int = 128) -> torch.Tensor:
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(model_name, device=str(dev))
    embeds = st.encode(sentences, convert_to_tensor=True, device=str(dev),
                        batch_size=batch_size, show_progress_bar=False)
    del st
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return F.normalize(embeds, dim=-1).cpu()


def retrieval_accuracy(adapter: nn.Module, a_embeds: torch.Tensor, b_embeds: torch.Tensor,
                        dev: torch.device) -> float:
    adapter.eval()
    with torch.no_grad():
        proj = F.normalize(adapter(a_embeds.to(dev)), dim=-1)
        logits = proj @ b_embeds.to(dev).T
        preds = logits.argmax(dim=-1)
        labels = torch.arange(a_embeds.size(0), device=dev)
        acc = (preds == labels).float().mean().item()
    adapter.train()
    return acc


def make_step_fn(adapter: nn.Module, optimizer: torch.optim.Optimizer,
                  a_train: torch.Tensor, b_train: torch.Tensor, dev: torch.device,
                  a_eval: torch.Tensor, b_eval: torch.Tensor) -> StepFn:
    n = a_train.size(0)

    def step_fn(step):
        idx = torch.randint(0, n, (BATCH_SIZE,))
        a_batch = a_train[idx].to(dev)
        b_batch = b_train[idx].to(dev)
        proj = F.normalize(adapter(a_batch), dim=-1)
        logits = proj @ b_batch.T / TEMPERATURE
        labels = torch.arange(a_batch.size(0), device=dev)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        quality = retrieval_accuracy(adapter, a_eval, b_eval, dev)
        return loss.item(), quality

    return step_fn


def main() -> RetrainHarness:
    dev = device()
    print(f"[text_text] device={dev}")

    print("[text_text] loading AG News sentences...")
    train_sents, test_sents = load_sentences()

    print("[text_text] encoding with module A (all-MiniLM-L6-v2, fixed)...")
    a_train = encode_all("all-MiniLM-L6-v2", train_sents, dev)
    a_test = encode_all("all-MiniLM-L6-v2", test_sents, dev)

    print("[text_text] encoding with module B v1 (paraphrase-MiniLM-L6-v2)...")
    b_train_v1 = encode_all("paraphrase-MiniLM-L6-v2", train_sents, dev)
    b_test_v1 = encode_all("paraphrase-MiniLM-L6-v2", test_sents, dev)

    dim_a = a_train.size(-1)
    dim_b = b_train_v1.size(-1)
    adapter = Adapter(dim_a, dim_b).to(dev)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)

    harness = RetrainHarness("text_text")

    print("[text_text] phase 1: initial adapter training (module B v1)...")
    step_fn = make_step_fn(adapter, optimizer, a_train, b_train_v1, dev, a_test, b_test_v1)
    rec1 = harness.run("initial_train", step_fn, max_steps=300, eval_every=5,
                        recover_target=None, patience_stop=None)
    quality_before_swap = rec1.history[-1]["quality"]
    rec1.quality_before_swap = quality_before_swap
    print(f"  -> reached quality={quality_before_swap:.3f}")

    print("[text_text] swapping module B: paraphrase-MiniLM-L6-v2 -> all-MiniLM-L12-v2 (\"index re-encoded with newer model\")...")
    b_train_v2 = encode_all("all-MiniLM-L12-v2", train_sents, dev)
    b_test_v2 = encode_all("all-MiniLM-L12-v2", test_sents, dev)

    quality_after_swap = retrieval_accuracy(adapter, a_test, b_test_v2, dev)
    print(f"  -> quality immediately after swap (adapter NOT retrained) = {quality_after_swap:.3f}")

    print("[text_text] phase 2: retraining adapter against module B v2...")
    step_fn2 = make_step_fn(adapter, optimizer, a_train, b_train_v2, dev, a_test, b_test_v2)
    rec2 = harness.run("post_swap_retrain", step_fn2, max_steps=300, eval_every=5,
                        recover_target=0.95 * quality_before_swap, patience_stop=20)
    rec2.quality_before_swap = quality_before_swap
    rec2.quality_immediately_after_swap = quality_after_swap

    path = harness.save(RESULTS_DIR)
    print(f"[text_text] saved -> {path}")
    print(f"[text_text] retrain_time_sec={rec2.retrain_time_sec}, steps_to_recover={rec2.steps_to_recover}")
    return harness


if __name__ == "__main__":
    main()
