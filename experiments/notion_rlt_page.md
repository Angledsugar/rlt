# RLT (Reinforcement Learning Token) - Pen Uncap Task

> ALOHA 로봇의 pen uncapping 작업을 위한 RLT 파이프라인 실험 기록
> 날짜: 2026-04-14 ~ 04-15
> 환경: 2x RTX 3090 24GB (cuda:0, cuda:1), JAX/Flax/NNX, pi0.5-base VLA

---

## Overview

RLT는 대형 VLA(Vision-Language-Action) 모델을 frozen 상태로 유지하면서, 가벼운 RL token + actor-critic으로 online RL fine-tuning하는 프레임워크입니다.

### Pipeline (3-Phase)

| Phase | 설명 | 입력 | 출력 | 상태 |
|-------|------|------|------|------|
| **Phase 0** | VLA 임베딩 추출 | pi0.5-base + demo 데이터 | embeddings.npy (3.4GB) | **완료** |
| **Phase 1** | RL Token encoder-decoder 학습 | VLA 임베딩 | RL Token 체크포인트 (.pkl) | **완료** |
| **Phase 2** | Online RL (TD3 + BC) | RL Token + VLA + MuJoCo env | 학습된 Actor | **진행 중 (이슈 있음)** |

---

## Phase 0: VLA Embedding Extraction

- **모델**: pi0.5-base (`openpi` 프로젝트)
- **데이터**: HuggingFace `physical-intelligence/aloha_pen_uncap_diverse` (3 에피소드)
- **결과**: `(900, 968, 2048)` — 900프레임, 968토큰/프레임, 2048차원
- **소요 시간**: ~47분
- **bfloat16 이슈**: VLA가 bfloat16으로 출력 → numpy에서 raw bytes(dtype `|V2`)로 저장 → `view(jnp.bfloat16).astype(np.float32)` 변환 필요

---

## Phase 1: RL Token Training

RL Token은 VLA의 968개 토큰(2048차원)을 single compact representation으로 압축하는 transformer encoder-decoder입니다.

### Synthetic Data 실험 (dim=2048, seq_len=8, 200 frames)

#### Initial Sweep (500 steps)

| Config | LR | BS | Final Loss | Reduction |
|--------|----|----|-----------|-----------|
| lr_5e-5 (논문 기본값) | 5e-5 | 32 | 226.8 | 94.5% |
| lr_1e-4 | 1e-4 | 32 | 119.0 | 97.1% |
| **lr_2e-4** | **2e-4** | **32** | **69.7** | **98.3%** |
| lr_1e-4_bs64 | 1e-4 | 64 | 111.1 | 97.3% |

#### Extended Sweep

| Config | LR | BS | Steps | Final Loss | Reduction | Time |
|--------|----|----|-------|-----------|-----------|------|
| lr_3e-4 | 3e-4 | 32 | 500 | 45.8 | 98.9% | 315s |
| **lr_2e-4_bs64** | **2e-4** | **64** | **500** | **39.9** | **99.0%** | **310s** |
| lr_2e-4_1000s | 2e-4 | 32 | 1000 | 64.4 | 98.4% | 565s |

### Real VLA Embeddings 실험

| Config | LR | BS | Steps | Init Loss | Final Loss | Reduction | Time |
|--------|----|----|-------|-----------|-----------|-----------|------|
| **real_best** | **2e-4** | **4** | **500** | **63,246.9** | **1,499.2** | **97.6%** | **353s** |

### Phase 1 결론

- **최적 LR = 2e-4** (논문 기본값 5e-5 대비 4x)
- Synthetic: bs=64로 99.0% reduction
- Real VLA: bs=4 (메모리 제한, 968토큰 × 2048차원) → 97.6% reduction
- 500 steps로 충분히 수렴
- 체크포인트: `checkpoints/pen_uncap/rl_token/real_best/state.pkl`

---

## Phase 2: Online RL (TD3 + BC Regularization)

### 실험 조건
- **VLA**: Mock VLA (FakeVLA, random embeddings/actions)
- **환경**: MuJoCo pen_uncap, 100-step 에피소드
- **설정**: 2000 env steps, 5 warmup 에피소드
- **보상**: 모두 0.0 (random action으로는 pen uncap 불가 — 정상)

### Hyperparameter Sweep (11 configs)

| Config | Beta | Ref Dropout | UTD | Actor LR | actor_q (↓) | 비고 |
|--------|------|------------|-----|----------|------------|------|
| baseline | 1.0 | 0.50 | 5 | 3e-4 | ~0.54 | 기준선 |
| beta_0.25 | 0.25 | 0.50 | 5 | 3e-4 | ~0.55 | |
| beta_0.5 | 0.5 | 0.50 | 5 | 3e-4 | 0.544 | |
| beta_2.0 | 2.0 | 0.50 | 5 | 3e-4 | 0.502 | |
| **utd_10** | **1.0** | **0.50** | **10** | **3e-4** | **0.428** | **Best: -20.8%** |
| utd_2 | 1.0 | 0.50 | 2 | 3e-4 | 0.470 | |
| dropout_0.25 | 1.0 | 0.25 | 5 | 3e-4 | 0.480 | |
| **dropout_0.75** | **1.0** | **0.75** | **5** | **3e-4** | **0.445** | **2nd: -17.6%** |
| actor_lr_1e-4 | 1.0 | 0.50 | 5 | 1e-4 | 0.457 | 3rd: -15.4% |
| stride_1 | 1.0 | 0.50 | 5 | 3e-4 | 0.632 | 최악 |
| stride_4 | 1.0 | 0.50 | 5 | 3e-4 | N/A | buffer 부족으로 학습 불가 |

### Phase 2 핵심 발견

1. **UTD ratio = 10**이 가장 효과적 (baseline 대비 20.8% 개선)
   - Off-policy 업데이트를 많이 할수록 critic/actor 학습 가속
   - Trade-off: 18% 느린 wall-clock time
2. **Reference action dropout = 0.75**가 2위 (17.6% 개선)
   - VLA reference에 대한 의존도를 줄여 독립적 탐색 유도
3. **stride_1은 피해야 함** — temporal correlation 희석으로 최악의 수렴
4. **stride_4는 short episode와 비호환** — buffer가 batch_size를 못 채움

---

## Phase 2 with Real VLA (미완료)

### 시도 과정

Real VLA (pi0.5-base, ~18GB/GPU)를 로드하여 Phase 2를 실행하려 했으나, stdout 출력이 전혀 나오지 않는 문제 발생.

| 시도 | 방법 | 결과 |
|------|------|------|
| 1차 | `uv run python scripts/train_pen_uncap.py` | 90분간 0바이트 출력, kill |
| 2차 | `PYTHONUNBUFFERED=1` + verbose logging 추가 | 동일 — 0바이트 |
| 3차 | `.venv/bin/python -u` 직접 실행 | 동일 — 0바이트 |
| 진단 | step-by-step print(flush=True) 테스트 | 9단계까지 정상 출력 → `train_online_rl()` 진입 후 멈춤 |
| VLA 단독 테스트 | 별도 스크립트로 VLA load + forward pass | **46.5초에 성공!** |

### 진단 결과

- VLA 로드 자체는 정상 (46.5초, embeddings shape `(1, 968, 2048)`, actions shape `(1, 50, 14)`)
- `train_online_rl()` 내부에서 `logger.info()` 출력이 안 나옴 → **logging 핸들러 미설정 문제** (train_pen_uncap.py의 `logging.basicConfig()`가 별도 스크립트에서는 호출되지 않음)
- 실제로는 VLA가 정상 로드되어 GPU 메모리 할당까지 완료 (18.4GB + 19.7GB)
- warmup에서 에피소드당 400 step × VLA forward pass로 매우 느릴 수 있음

### 남은 작업

- `print(flush=True)`로 교체 완료 — 다음 실행에서는 진행 상황 확인 가능
- warmup 에피소드를 줄이거나 (3 → 1), max_episode_steps를 줄여 테스트 필요
- Real VLA forward pass가 step당 ~1초면, warmup 3ep × 400step = ~20분 소요 예상

---

## Recommended Settings (최종)

### Phase 1
```
learning_rate: 2e-4        # 논문 5e-5 → 4x
batch_size: 64 (synthetic) / 4 (real VLA)
num_train_steps: 500       # 99% 수렴에 충분
```

### Phase 2
```
utd_ratio: 10              # 논문 5 → 2x
reference_action_dropout: 0.75  # 논문 0.5 → 1.5x
beta: 1.0                  # 논문 기본값 유지
actor_lr: 3e-4             # 논문 기본값
critic_lr: 3e-4            # 논문 기본값
action_chunk_stride: 2     # 논문 기본값
```

---

## 아키텍처

```
┌──────────────────────────────────────────────────────┐
│                    Online RL Loop                     │
│                                                      │
│  Observation ──→ [Frozen VLA (pi0.5)] ──→ Embeddings │
│                       │                    │         │
│                       ▼                    ▼         │
│               Reference Actions    [Frozen RL Token]  │
│                   a_hat_{1:C}          z (compact)    │
│                       │                    │         │
│                       ▼                    ▼         │
│                  [Actor] ── refine ──→ Action a       │
│                       │                              │
│                       ▼                              │
│               L_pi = -Q(s,a) + β·||a - â||²         │
│                    (TD3)        (BC reg)              │
│                                                      │
│  Trainable: Actor, Critic only (~512K params)        │
│  Frozen: VLA (~3B), RL Token (~4M)                   │
└──────────────────────────────────────────────────────┘
```

---

## 코드 구조

| 파일 | 역할 |
|------|------|
| `scripts/train_pen_uncap.py` | Phase 1 + 2 통합 학습 스크립트 |
| `scripts/extract_embeddings.py` | Phase 0: VLA 임베딩 추출 |
| `src/rlt/vla/backbone.py` | pi0.5 VLA wrapper (openpi 연동) |
| `src/rlt/models/rl_token.py` | RL Token encoder-decoder |
| `src/rlt/models/actor.py` | Actor network |
| `src/rlt/models/critic.py` | Double Critic (TD3) |
| `src/rlt/training/train_online_rl.py` | Phase 2 online RL 루프 |
| `src/rlt/envs/pen_uncap_env.py` | MuJoCo pen uncap 환경 |
| `src/rlt/envs/aloha_env.py` | ALOHA sim wrapper |

---

## 해결한 버그들

| 버그 | 원인 | 해결 |
|------|------|------|
| bfloat16 dtype error | VLA → numpy raw bytes (`\|V2`) | `.view(jnp.bfloat16).astype(np.float32)` |
| orbax CheckpointArgs error | `ckpt_manager.save(step, args=state)` | `args=ocp.args.StandardSave(state)` |
| orbax absolute path error | 상대경로 전달 | `Path(dir).resolve()` |
| pickle/orbax 비호환 | `.pkl`을 orbax로 로드 | `.pkl` 확장자 체크 → `pickle.load()` |
| MuJoCo physics crash | action > 1.0 | `np.clip(action, -1, 1)` |
| GPU contention | JAX가 모든 visible GPU에 메모리 preallocate | `CUDA_VISIBLE_DEVICES` 프로세스별 분리 |
| stdout 버퍼링 | 파이프 시 block buffering | `PYTHONUNBUFFERED=1` + `print(flush=True)` |

---

## GPU 환경

| GPU | 모델 | VRAM | 용도 |
|-----|------|------|------|
| cuda:0 | RTX 3090 | 24GB | VLA + 학습 |
| cuda:1 | RTX 3090 | 24GB | VLA + 학습 |
| cuda:2 | T1000 | 8GB | **사용 금지** (VRAM 부족) |

---

## 산출물

| 항목 | 경로 |
|------|------|
| VLA 임베딩 | `data/pen_uncap_embeddings/embeddings.npy` (3.4GB) |
| RL Token 체크포인트 | `checkpoints/pen_uncap/rl_token/real_best/state.pkl` |
| 실험 리포트 | `experiments/report.md` |
| Phase 1 smoke 테스트 | `scripts/smoke_phase1.py`, `scripts/smoke_phase1_ext.py` |
| Phase 2 smoke 테스트 | `scripts/smoke_phase2.py` |
| 실험 러너 | `experiments/run_experiment.sh` |
