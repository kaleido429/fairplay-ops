# Fairplay Fairy 파이프라인

영상 폴더에서 치트 판정 결과까지 한 번에 실행하는 파이프라인. 추출한 값은 SQLite에 저장하고, 집계와 판정과 분석은 SQL로 한다.

```
영상(.webm) ──ingest──> videos
            ──extract─> frames     GPU Optical Flow (Farneback), 프레임별 dx, dy
            ──score───> windows    윈도우 특징 105개 → XGBoost 확률 (Python)
                        thresholds 임계값
            ──report──> sql/*.sql  영상 점수, 판정, 재현율·오탐률, 오탐 분석 (SQL)
```

## 실행

```
python run_pipeline.py --db fairplay.db --train <학습 영상 폴더> --holdout <홀드아웃 영상 폴더>
python run_pipeline.py --db fairplay.db --stages report
```

- 영상 폴더 구조: `<폴더>/cheats/*.webm`(치트), `<폴더>/videos/*.webm`(정상)
- `--backend gpu|cpu|auto`: 기본 auto. CUDA로 빌드한 OpenCV가 있으면 GPU 사용. 위치는 환경변수 `OPENCV_CUDA_DIR`.
- extract 단계는 이미 처리한 영상을 건너뛴다. 중간에 끊겨도 다시 실행하면 이어서 처리한다.
- 필요 패키지: numpy, pandas, scipy, scikit-learn, xgboost, OpenCV

## 구성

| 파일 | 내용 |
|---|---|
| `schema.sql` | 테이블 5개(videos, frames, windows, thresholds, runs)와 뷰 4개 |
| `flow.py` | 영상 → 프레임별 조준 이동량. GPU / CPU |
| `features.py` | 프레임 특징 7개, 윈도우 특징 105개, 모델 |
| `run_pipeline.py` | 단계 실행 |
| `sql/01~06` | 분석 쿼리 |

뷰

| 뷰 | 내용 |
|---|---|
| `frame_features` | 프레임별 속도, 가속도, 저크, 각도, 각도 변화 (LAG 사용) |
| `video_scores` | 영상별 윈도우 확률의 최댓값과 평균 |
| `verdicts` | 영상 점수와 임계값을 비교한 판정 |
| `video_activity` | 영상별 평균 속도와 정지 프레임 비율 |

## 실행 결과 (2026-11-27, 영상 347개)

- 학습 256개(치트 25, 정상 231), 홀드아웃 91개(치트 43, 정상 48). 프레임 273,300개, 윈도우 7,466개.
- 학습 정상 영상 1개는 68프레임뿐이라 윈도우가 없어 학습과 판정에서 빠진다.
- 전체 1,875초: GPU 추출 1,708초(6개 프로세스), 특징·학습·임계값 163초.

홀드아웃 (치트 43, 정상 48)

| 규칙 | 임계값 방식 | 임계값 | 적발 | 오탐 |
|---|---|---|---|---|
| 윈도우 최댓값 | 분할 1회 | 0.99972 | 35 | 0 |
| 윈도우 최댓값 | 분할 20회 평균 | 0.99969 | 36 | 0 |
| 영상 평균 | 분할 1회 | 0.57643 | 38 | 4 |
| 영상 평균 | 분할 20회 평균 | 0.63644 | 33 | 4 |

- 이 홀드아웃은 방법을 정하는 과정에서 이미 여러 번 확인했다. 위 수치는 새 데이터에서의 기대 성능이 아니라 파이프라인이 앞선 분석과 같은 값을 내는지 확인한 것이다.
- 오탐 4건은 모두 평균 속도 하위 25% 영상이다(`sql/04_activity_groups.sql`). 3건은 정지 프레임 비율이 약 50%다.

## 확인한 것

- `frames`의 값이 별도 스크립트로 추출한 GPU 결과와 완전히 일치 (홀드아웃 69,582프레임).
- `frame_features` 뷰(SQL)와 `features.per_frame`(Python)의 값 차이 1e-13 이하 (영상 3개).
- SQL로 구한 학습 정상 영상 최고 점수가 Python이 저장한 임계값과 일치 (`sql/06_threshold_check.sql`).

## 관련 문서

- `../notes/cpu_gpu_optical_flow_검증.md`: CPU와 GPU의 결과 차이, GPU 재학습
- `../notes/임계값_결정방법_검토.md`: 임계값이 분할 시드에 좌우되는 문제와 평균 방식
