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

## 다음 단계 (②번, 아직 시작 안 함)
ML 레벨 실험: 실제로 독립적으로 사전학습된 소형 모델 2개를 준비하고, 그 사이를
잇는 작은 어댑터(LoRA류 또는 LLaVA식 프로젝션 레이어)를 학습시켜서:
- 어댑터 학습에 실제로 얼마나 걸리는지 (위 시뮬레이션의 retrain_time 상수의 근거)
- 학습 도중/직후 품질이 얼마나 떨어졌다가 회복하는지 곡선의 실제 모양
- 한쪽 모델을 다른 버전으로 교체했을 때 어댑터가 얼마나 망가지는지
를 측정해서 시스템 시뮬레이션의 가정들을 검증하는 것이 목표.

## 실행 방법 (다른 PC에서)
```
cd ai-modular-sim
python -m venv venv
# Windows:
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe main.py
```

## 새 세션에서 이어가려면
이 CONTEXT.md 전체를 Claude에게 붙여넣고 "여기서부터 이어서, ② ML 레벨 실험을
시작해줘" 라고 요청하면 됨.
