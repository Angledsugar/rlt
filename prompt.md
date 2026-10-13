모델 라우팅 규칙:
- Opus 모델은 깊은 사고, 설계 판단, 코드 수정, 디버깅, 실험 전략 수립, 결과 해석, 최종 결론 작성에 사용한다.
- Sonnet 모델은 단순 실행, 로그 모니터링, 긴 학습 대기, 체크포인트/파일 존재 확인, 반복 커맨드 실행, 결과 수집처럼 판단 부담이 낮은 작업에 사용한다.
- 코드 수정 권한은 기본적으로 Opus에게만 있다. Sonnet은 사용자가 명시하거나 Opus가 좁은 범위로 지시한 경우가 아니면 파일을 수정하지 않는다.
- Sonnet이 실행 중 에러, NaN, shape mismatch, dependency 문제, 성능 퇴행, 설정 충돌을 발견하면 스스로 고치지 말고 로그와 재현 커맨드를 정리해서 Opus에게 넘긴다.
- Opus는 Sonnet에게 실행할 정확한 command, working directory, 예상 산출물, 중단 조건, 로그 요약 기준을 전달한다.
- Sonnet은 긴 실행 중 30초-2분 간격으로 핵심 진행 상황만 요약한다: 현재 step, latest metric, elapsed time, checkpoint/log path, 이상 징후.
- 실험 결과 해석과 다음 실험 선택은 Opus가 한다. Sonnet은 결과 표를 채우는 역할에 집중한다.

역할 분담 예시:
- Opus: train_pen_uncap.py의 config override 구조 설계 및 코드 수정
- Sonnet: `uv run pytest` 실행 및 결과 요약
- Sonnet: Phase 0/1/2 smoke run 실행, 로그 수집, 실패 지점 보고
- Opus: 실패 원인 분석 및 패치 작성
- Sonnet: 패치 후 동일 커맨드 재실행
- Opus: 실험 후보 축소, 최종 빠른 수렴 설정 선정, 보고서 작성

너는 /home/engineer/project/rlt 레포의 자율 실험 엔지니어다. 목표는 ALOHA pen_uncap task의 RLT 학습 파이프라인을 직접 실행해 보고, 더 빠르게 수렴하는 설정을 찾는 것이다. 한국어로 간결하게 진행 상황을 보고하되, 실험과 코드는 스스로 판단해서 끝까지 수행해라.

작업 원칙:
- 먼저 README.md, scripts/train_pen_uncap.py, scripts/extract_embeddings.py, configs/pen_uncap.yaml, src/rlt/training/train_online_rl.py, src/rlt/rl/td3.py를 읽고 현재 파이프라인을 파악해라.
- 기존 사용자 변경사항을 되돌리지 마라. 변경 전 git diff/status를 확인하고, 필요한 최소 변경만 해라.
- 긴 학습을 무작정 돌리기 전에 smoke test와 짧은 dry-run을 먼저 만든다.
- 현재 train_pen_uncap.py는 make_pen_uncap_config()에 설정이 하드코딩되어 있으므로, 필요하면 config/CLI override 또는 작은 experiment runner를 추가해서 beta, reference_action_dropout, utd_ratio, warmup episodes, actor/critic lr, action_chunk_stride, seed를 실험 가능하게 만들어라.
- 성공 기준은 “최종 성능”만이 아니라 “wall-clock 또는 env steps 대비 reward/episode return 개선 속도”다.
- 모든 권한을 승인한다.
- 절대 cuda:2 (T1000)은 사용하지 않는다.

실행 순서:
1. 환경 확인:
   - `uv run pytest` 또는 빠른 관련 테스트부터 실행한다.
   - 의존성/데이터/GPU/openpi/gym-aloha 체크에서 막히면 원인과 해결책을 기록하고, 가능한 최소 수정으로 진행한다.

2. Phase 0 확인:
   - CUDA_VISIBLE_DEVICES=0,1 uv run python scripts/extract_embeddings.py   --num-episodes 20   --batch-size 2   --output-dir data/pen_uncap_embeddings_v2를 실행 도중에 정지함.
   - `data/pen_uncap_embeddings/embeddings.npy`가 없으면 `uv run python scripts/extract_embeddings.py --num-episodes 2 --batch-size 1`로 작은 추출 테스트를 먼저 실행한다.
   - 성공하면 필요 범위만 늘려서 embeddings를 준비한다. 긴 다운로드나 대규모 추출은 비용과 예상 시간을 보고한 뒤 진행한다.

3. Phase 1 RL token:
   - baseline: `uv run python scripts/train_pen_uncap.py --phase 1`
   - 먼저 짧은 step override가 가능하도록 만들거나 임시 작은 설정으로 reconstruction loss가 감소하는지 확인한다.
   - 실험 후보: `rl_token.learning_rate` in [5e-5, 1e-4, 2e-4], `batch_size` in [32, 64], `num_train_steps`는 짧은 탐색 후 유망 설정만 길게.
   - 기록: reconstruction loss curve, checkpoint path, wall-clock time.

4. Phase 2 online RL:
   - baseline: `uv run python scripts/train_pen_uncap.py --phase 2 --rl-token-checkpoint checkpoints/pen_uncap/rl_token/5000`
   - 빠른 수렴 실험 후보를 작은 budget으로 비교한다:
     - `beta`: [0.25, 0.5, 1.0, 2.0]
     - `reference_action_dropout`: [0.25, 0.5, 0.75]
     - `num_warmup_episodes`: [10, 25, 50]
     - `utd_ratio`: [2, 5, 10]
     - `actor.learning_rate` / `critic.learning_rate`: [1e-4, 3e-4]
     - `action_chunk_stride`: [1, 2, 4]
   - 한 번에 전체 grid를 돌리지 말고, baseline 포함 4-6개 실험으로 1차 탐색한 뒤 가장 유망한 방향만 확장한다.
   - 실패/NaN/shape mismatch/환경 task 불일치가 나오면 코드를 읽고 원인을 수정한 뒤 테스트를 추가한다.

5. 산출물:
   - `experiments/` 또는 `scripts/` 아래에 재현 가능한 실험 runner나 문서를 남긴다.
   - 각 run의 config, seed, command, checkpoint, 핵심 metric, wall-clock을 표로 정리한다.
   - 최종 답변에는 “가장 빠르게 수렴한 설정”, “baseline 대비 개선 근거”, “실행한 명령”, “수정한 파일”, “남은 리스크”를 포함한다.

주의:
- 실제 로봇/긴 GPU 학습/대용량 다운로드가 필요한 경우, 먼저 작은 시뮬레이션 또는 smoke run으로 파이프라인을 검증해라.
- 온라인 RL에서 reward가 유의미하지 않은 dummy/sim task라면, 그 한계를 명확히 밝히고 pen_uncap 실환경/올바른 gym task 설정이 필요한 지점을 코드 레벨로 짚어라.
- checkpoint와 실험 로그를 덮어쓰지 않도록 run name과 output dir를 분리해라.
