# ai-modular-sim — 프로젝트 컨텍스트

## 연구 배경
장기 목표: AI 전용 칩셋을 만들어서, 칩 하나에 모듈화된 모델 하나씩을 이식하고,
모델을 업데이트할 때 그 칩만 교체하는 방식의 시스템을 연구하는 것 (컴퓨터가
한 기판에서 CPU/GPU/RAM이 분리된 부품 구조로 진화한 것과 같은 비유를 AI에 적용).

논의 중 정리된 핵심 포인트:
- 대형 모델의 MoE(Mixture-of-Experts) 구조는 "여러 작은 모델의 조합"처럼 보이지만,
  실제로는 처음부터 하나의 손실함수로 end-to-end 함께 학습되어 서로 표현을 공유함.
  독립된 모듈처럼 분리해서 재사용하기 어려움.
- CLIP+LLM(LLaVA류) 같은 모델은 실제로 "모듈 + 인터페이스(프로젝션 레이어)" 구조를
  이미 구현한 사례. 단, 인터페이스 자체도 학습이 필요하고, 한쪽 모듈이 크게
  바뀌면 인터페이스도 재학습해야 함 (완전한 플러그 앤 플레이는 아직 불가능).
- 하드웨어 구현 경로: (1) 범용 칩 + 분산 배치(MoE expert parallelism, 이미 실무에서 사용중),
  (2) 아날로그/인메모리 컴퓨팅(멤리스터 크로스바 등, 가중치를 물리적으로 새겨넣는 방식,
  Mythic/IBM 아날로그 AI 칩 등) — 이게 사용자 아이디어와 가장 가까운 실현체지만
  업데이트=재제작이라는 트레이드오프가 있음.
- 실제 반도체 제작은 자본/공정 장벽이 커서, 우선 **시뮬레이션으로 타당성 검증**부터
  하기로 함. 순서: ① 시스템/아키텍처 레벨 시뮬레이션 → ② ML 레벨 실험(실제 소형
  모델 2개 + 학습된 어댑터로 인터페이스 재학습 현상 검증).

## 지금까지 구현한 것 (①번, 시스템 레벨 시뮬레이션 — 완료)

Python + simpy(discrete-event simulation) 기반. 폴더 구조:

```
ai-modular-sim/
  venv/                  # Python 가상환경 (다른 PC에서는 재생성 필요, requirements.txt 참고)
  sim/
    core.py       # Module, Interface, RetrainState, Pipeline 클래스
    metrics.py    # MetricsCollector — 요청별 (시간, latency, quality) 기록 및 시간대별 집계
    scenario.py   # 모놀리식/모듈러 파이프라인 구성, 트래픽 생성기, 업데이트 스케줄러
    plot.py       # 결과 그래프 + 텍스트 요약 출력
  main.py         # 엔트리포인트 (python main.py 실행)
  results.png     # 마지막 실행 결과 그래프
  requirements.txt
  CONTEXT.md      # 이 파일
```

### 핵심 설계
- **Module** = "칩" 하나. base_latency(처리시간), capacity(동시처리 슬롯 수), 그리고
  RetrainState(교체 시 품질이 얼마나 떨어지고 얼마나 걸려 회복하는지)를 가짐.
- **Interface** = 두 Module을 잇는 학습된 변환 레이어("어댑터"). comm_latency +
  translation_cost, 그리고 자신만의 RetrainState를 가짐.
- **모놀리식 아키텍처** = Module 1개, Interface 0개. 업데이트 시 그 Module 자체가
  "전체 재학습" 상태로 들어감 (retrain_time=150, penalty_factor=25x — 매우 느려짐).
- **모듈러 아키텍처** = Module 3개 + Interface 2개 체인. 업데이트 시 해당 Module에
  인접한 Interface만 재학습 상태로 들어감 (retrain_time=40, penalty_factor=3x —
  상대적으로 가벼움). Module 자체는 즉시 정상 동작 (hot-swappable 가정).
- 두 아키텍처 모두 같은 Poisson 트래픽(mean_interarrival=4)과 t=200/500/800 업데이트
  이벤트를 받음. MetricsCollector가 시간대별(10 time-unit bucket) latency/throughput/
  quality를 기록.

### 실행 결과 (results.png) — 핵심 발견
- **모놀리식**: 업데이트 순간 처리 지연이 25배로 뛰면서, capacity=4인 자원 큐에 요청이
  쌓이기 시작함. 서비스 속도가 도착 속도보다 느려지는 순간 큐잉이 무한정 누적되는
  현상이 발생 — t=230 이후 처리량이 사실상 0으로 붕괴, t=800 업데이트는 시뮬레이션이
  끝날 때까지 발동조차 못 함 (이전 백로그를 처리하느라). 즉 "일시적 성능 저하"가
  아니라 **큐 폭주로 인한 사실상 영구 장애**.
- **모듈러**: 업데이트마다 quality가 0.5~0.6까지 떨어지지만 40~60 time-unit 안에
  1.0으로 회복. latency/throughput은 거의 영향 없음. 세 번의 업데이트 모두 동일한
  패턴으로 회복.
- **결론**: 변화의 영향 범위를 격리할 수 있느냐가 핵심 — 모듈화가 구조적으로 유리함.
  단, retrain_time/penalty_factor/degraded_quality 값은 그럴듯하게 지정한 상수이며
  실측 근거는 없음. 정성적 패턴(격리 > 전체 재학습)은 robust하지만 정확한 숫자는
  다음 단계에서 검증 필요.

## ② ML 레벨 실험 (완료 — 1차 라운드)

`ml_experiment/` 폴더에 두 개의 실제 실험을 구현/실행함. 둘 다 "module A(고정) +
module B(고정) + 학습 가능한 Adapter" 구조에서, 처음부터 adapter를 학습시킨 뒤
한쪽 module을 교체하고 adapter quality가 얼마나 깨지는지/얼마나 빨리 회복하는지
측정하는 동일한 프로토콜(`ml_experiment/common.py`의 `RetrainHarness`)을 공유.

폴더 구조:
```
ml_experiment/
  common.py           # Adapter(2-layer MLP+residual), RetrainHarness (학습/재학습 곡선 기록)
  exp_image_text.py   # 실험 1: 미니 CLIP류 (이미지<->텍스트)
  exp_text_text.py    # 실험 2: 문장 임베딩 공간 stitching (텍스트<->텍스트)
  plot_results.py     # results/*.json -> quality_curves.png + summary.txt
  run_all.py          # 두 실험 순차 실행 + plot
  requirements.txt    # sentence-transformers, datasets, tqdm
                      # (torch/torchvision은 GPU판이라 별도: pip install torch torchvision
                      #  --index-url https://download.pytorch.org/whl/cu121)
  results/            # *.json (raw curve), quality_curves.png, summary.txt
```

### 실험 1: 미니 CLIP (이미지<->텍스트)
- Module A = frozen 이미지 인코더(ResNet18 ImageNet-pretrained) → 나중에 ResNet34로 교체.
- Module B = frozen 텍스트 인코더(sentence-transformers `all-MiniLM-L6-v2`), 고정.
- Interface = adapter가 이미지 임베딩을 텍스트 임베딩 공간으로 projection.
- Task: CIFAR-10 이미지를 "a photo of a {class}" 프롬프트 10개 중 하나로 분류
  (zero-shot 방식), quality = top-1 accuracy. 이미지 임베딩은 미리 캐싱해서
  adapter 학습 자체는 매우 빠름(순수 MLP 연산).
- **결과**: 초기 학습 83.9% 정확도 도달 → ResNet18→ResNet34 교체 직후 **7.9%로 붕괴**
  (거의 랜덤 수준, 인터페이스 완전 파괴) → adapter만 재학습 시 **20 스텝(0.13초)** 만에
  95% 수준(79.7%) 회복, 최종적으로는 원래보다 살짝 높은 87%대까지 회복.

### 실험 2: 텍스트<->텍스트 임베딩 공간 stitching
- Module A = frozen 문장 임베딩 모델(`all-MiniLM-L6-v2`), 고정.
- Module B = frozen 문장 임베딩 모델, 처음엔 `paraphrase-MiniLM-L6-v2` → 나중에
  `all-MiniLM-L12-v2`로 교체("검색 인덱스가 최신 임베딩 모델로 재인코딩됨" 시나리오).
- Interface = adapter가 A의 임베딩을 B의 임베딩 공간으로 translation. Label 없이
  같은 문장의 A/B 임베딩 쌍만으로 in-batch contrastive 학습(자기지도).
- Task: AG News 문장(`fancyzhx/ag_news`) 3000개로 학습, 800개로 평가. quality =
  batch 내에서 자기 자신의 B 임베딩을 negative들 사이에서 찾아내는 top-1 retrieval accuracy.
- **결과**: 초기 학습 96.9% → module B 교체 직후 **95.4%** (드롭 단 1.5%) →
  5 스텝(0.026초)만에 95% 회복 기준 통과.

### 핵심 발견 — 시뮬레이션 가정과의 비교
- **"module이 얼마나 다르게 바뀌었는가"가 interface 붕괴 폭을 결정한다**: 완전히 다른
  아키텍처로 교체(ResNet18→34, 서로 다른 학습 목적함수로 나온 이미지 vs 텍스트
  임베딩 정렬)는 거의 랜덤 수준까지 붕괴(-90.6%p). 반면 같은 계열의 문장 임베딩
  모델끼리 교체(MiniLM 계열, 비슷한 학습 레시피)는 거의 영향 없음(-1.5%p). 즉
  1차 시뮬레이션이 가정한 "고정된 degraded_quality 상수"는 과도하게 단순화된
  것 — 실제로는 module 간 표현 공간의 유사도에 따라 붕괴 폭이 크게 달라짐.
- **retrain_time은 시뮬레이션 가정(40 time-unit, 상대적으로 느림)과 질적으로 다른
  양상**: adapter 자체가 작고 임베딩을 캐싱해둔 상태에서는 재학습이 사실상
  즉각적(0.03~0.13초, 5~20 스텝)임. 시뮬레이션의 "재학습에 시간이 걸린다"는
  가정의 실제 병목은 gradient step 수가 아니라 ① 새 module로 전체 데이터셋의
  임베딩을 다시 뽑는 비용, ② 실제 서비스에 새 adapter를 배포하는 절차일 가능성이
  높다 — 다음 라운드에서 이 두 가지를 시간 축에 넣어 재검증할 필요.
- **정성적 결론(모듈화가 유리하다)은 여전히 유효**: 두 실험 모두 "작은 adapter만
  재학습하면 됨"이 실제로 매우 빠르고 값싸다는 것을 확인. monolithic처럼 전체
  모델을 재학습해야 하는 상황과 비교하면 여전히 압도적으로 유리함.

### 다음 라운드 후보
- retrain_time에 "새 module로 임베딩 재추출" 비용을 포함시켜 실측.
- module 간 표현 공간 유사도(예: CKA, 임베딩 공간 정렬도)를 직접 측정해서
  degraded_quality를 예측하는 지표로 쓸 수 있는지 확인.
- adapter 크기/구조를 바꿔가며 재학습 속도-품질 트레이드오프 스윕.

### 실행 방법
```
cd ml_experiment
../venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
../venv/Scripts/python.exe -m pip install -r requirements.txt
../venv/Scripts/python.exe run_all.py
```

## 실행 방법 (다른 PC에서)
```
cd ai-modular-sim
python -m venv venv
# Windows:
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe main.py
```

## 새 세션에서 이어가려면
이 CONTEXT.md 전체를 Claude에게 붙여넣고 "여기서부터 이어서, ② ML 레벨 실험
다음 라운드(임베딩 재추출 비용 포함한 retrain_time 재측정, 표현 공간 유사도
지표로 degraded_quality 예측) 진행해줘" 라고 요청하면 됨.
