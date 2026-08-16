# LG Aimers 9기 Phase2 — 투구 제구 성공 확률 예측

[대회 페이지](https://dacon.io/competitions/official/236743) · 리더보드 제출 마감 **2026-09-01** · 코드+PPT 마감 2026-09-07

## 현재 상태

| 항목 | 값 |
|---|---|
| 리더보드 Public | **897.09** (수료 커트 549.51, 운영진 베이스라인 대비 1.63배 스킬) |
| 2024 홀드아웃 (검증용 프록시) | 710.1 |
| 추론 시간 | 12.8초 / 245,789행 (제한 10분) |
| 제출 zip | 1.29 MB (제한 10GB) |

## 이 문제의 핵심

**`train.csv`는 2019~2024, 평가 데이터는 2025다.** 제구 성공률이 매년 떨어진다.

| 시즌 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|
| 성공률 | .5647 | .5327 | .5328 | .5289 | .5000 | .4861 |

평가식이 Brier Skill Score라 **예측 평균이 0.01 어긋나면 약 40점**이 날아간다. 상수 예측은 정확히 0점,
과적합하면 음수라 0점. 즉 정확도 경쟁이 아니라 **드리프트 보정 + 캘리브레이션** 경쟁이다.

대응: 학습 데이터의 시즌별 성공률에 선형추세를 적용해 2025를 외삽하고(0.4747), 그 차이를
**logit 공간의 고정 상수**로 모델에 저장한다. 평가 데이터 통계를 일절 쓰지 않으므로
"각 행은 독립적으로 예측" 규칙에 안전하다. 2024 홀드아웃에서 657.6 → 710.1 (오라클 상한 714.2).

## 실험에서 확인된 것

- **모델 용량을 키우면 나빠진다.** iter 800 → 565, leaf 63 → 482. 강한 정규화가 정답
  (`max_leaf_nodes=15, l2=10, min_samples_leaf=10000`) + 시드 4개 앙상블.
- **최근 시즌 가중치는 손해** (반감기 2시즌: -90점). 데이터 양이 이긴다.
- **Trackman은 붙일 수 없다.** `pitcher_trackman_id`(50008~71775155)와 메인 `pitcher_id`(20700~24633)의
  교집합이 0개. 익명화 체계가 완전히 분리돼 있다. 그리고 쓸 만한 신호는 이미 운영진이
  `asof_pitcher_fastball_rate` 등으로 뽑아 넣어놨다.
- 파생 피처 기여도 (2024 홀드아웃, 2시드): 상황 +37 > 카운트 +14 > asof 스무딩 −33(단독). 전부 합치면 757.

## 환경 함정

로컬 numpy는 2.x인데 **평가 서버는 numpy 1.26.4**다. numpy 2.x로 만든 pickle은
서버에서 `numpy._core` ImportError로 **로드 자체가 실패**한다. 반드시 `.venv-submit`에서 학습·저장할 것.

```bash
python -m venv .venv-submit
.venv-submit/Scripts/python.exe -m pip install "numpy==1.26.4" "scikit-learn==1.8.0" "joblib==1.5.3" "pandas==2.2.3"
```

`requirements.txt`는 주석 한 줄만 둔다 — 서버 기본 설치 패키지만 쓰면 설치 오류 위험이 0이다.

## 구조

```
open/                  대회 배포본 (data/ 는 git 제외, 689MB)
work/
  features.py          학습·추론이 공유하는 전처리 (범주형 매핑 고정)
  train.py             학습 → submit/model/model.joblib
  experiments/         홀드아웃 실험 기록 (01~06)
submit/                제출 zip의 내용물
  model/model.joblib
  script.py            평가 서버가 실행하는 추론 코드
  requirements.txt
```

## 사용법

```bash
# 2024 홀드아웃으로 절차 전체 검증
cd work && ../.venv-submit/Scripts/python.exe train.py --validate

# 전체 학습 후 저장
cd work && ../.venv-submit/Scripts/python.exe train.py

# 제출 zip 생성 (최상위 폴더가 한 겹 더 생기면 구조 불일치 오류가 난다)
cd submit && python -c "import zipfile; z=zipfile.ZipFile('../submit.zip','w',zipfile.ZIP_DEFLATED); [z.write(f,f) for f in ['script.py','requirements.txt','model/model.joblib']]; z.close()"
```

## 규칙 메모

- 1일 제출 5회. **설치 오류는 차감 없음, `script.py` 실행 중 오류는 차감됨.**
- 외부 데이터 금지, 외부 API 금지, 2025년 Trackman 금지.
- 평가 데이터 전체 통계를 이용한 사후 보정 금지 → 리더보드 피드백으로 상수를 튜닝하는 것도
  같은 조항에 걸린다. 보정값은 반드시 학습 데이터에서만 유도할 것.
