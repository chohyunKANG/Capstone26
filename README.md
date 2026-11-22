# Enkidu 음성 보호 × De-AntiFake 정화 × XTTS-v2 복제

음성 복제 방어 섭동 **Enkidu**가 학습형 정화 공격 **De-AntiFake**를 거친 뒤에도 제로샷 복제기 **XTTS-v2**를 막는지 측정하고, 사용성(명료도)과 보호를 함께 높이려는 개조를 시험한 기록이다.

정보보호관리및기술 캡스톤 · 팀 버그토벌대. 같은 실험 틀로 SceneGuard, SafeSpeech를 평가한 팀원 작업과 비교했다.

## 실험 설계

```
D1 원본 ─────────────────────────────► XTTS-v2 ─► C1 ┐
D1 ─► Enkidu ─► D2 보호본 ────────────► XTTS-v2 ─► C2 ├─► D1 원본과 화자 유사도
D2 ─► De-AntiFake ─► D3 정화본 ───────► XTTS-v2 ─► C3 │   (ECAPA, Resemblyzer)
D1 ─► De-AntiFake ─► D4 정화만(대조군) ► XTTS-v2 ─► C4 ┘
```

- **C1 − C2** 보호 효과, **C4 − C3** 정화 후 남은 보호 효과. `C1 − C3 = (C1 − C4) + (C4 − C3)`로 정화기 자체 손상과 보호 잔존을 분리한다.
- 임계값은 논문 기본값 대신 D1 동일 화자 쌍 475개로 직접 구한 **EER: ECAPA 0.323, Resemblyzer 0.739** (FAR 5%: 0.278 / 0.690).
- 데이터: LibriSpeech test-clean 화자 4명(1089, 1188, 121, 1221), 206개 발화. 화자별 70%로 패치를 학습하고 나머지 **63개**로 평가한다.

## 버전

| 버전 | 바꾼 것 | 상태 | 핵심 결과 |
|---|---|---|---|
| **v0** 원본 Enkidu | — | 완료 | 정화 후 잔존 C4−C3 = 0.265 (ECAPA), STOI 0.68 |
| **v1** 청각 마스킹 배치 | 빈·프레임별 상한 κ·√θ(f,t)로 섭동을 자름. 학습 단계에도 적용 | 완료 | STOI 0.87로 개선, 잔존 0.011 (p=0.48) → **trade-off 못 벗어남** |
| **v2** XTTS 대리 모델 | ECAPA 대신 XTTS-v2의 화자 임베딩·GPT 조건값을 흔들도록 학습 | 완료 | 같은 STOI에서 보호 효과 2배(대조군 대비), 정화 후 잔존 0 |
| **대조군** | 원본 패치를 크기만 0.25배 (STOI 0.873) | 완료 | 정화 후 잔존 0 → 원인은 **섭동 크기** |

자세한 수치는 [`results/README.md`](results/README.md).

## 폴더

```
notebooks/
  01_protect_enkidu.ipynb       D1 → D2 (패치 학습, 보호본 생성, 입력 단계 검증)
  02_purify_deantifake.ipynb    D2 → D3, D1 → D4 (De-AntiFake / PhonePuRe)
  03_clone_xtts.ipynb           D1~D4 → C1~C4, ECAPA·Resemblyzer 채점, EER, 쌍체 분석
variants/
  v1_masking/                   01_protect_v1 · 02_purify_v1 · 03_clone_v1 + 마스킹 모듈
  v2_xtts_surrogate/            01_protect_v2 · 02_purify_v2 · 03_clone_v2 + xtts_ufp.py
  ctrl_uniform/                 01_protect_ctrl · 02_purify_ctrl · 03_clone_ctrl (대조군)
results/README.md               전체 결과표와 코드 분석에서 발견한 점
```

## 실행

모두 Colab(T4 GPU)과 구글 드라이브 `MyDrive/xtts_exp/` 기준이다.

```
xtts_exp/
  D1/<화자>/libri_p<화자>-<번호>.wav (+ .txt 전사)   원본
  D2/ D3/ D4_new/                                      v0 산출물
  runs/run1/enkidu_ufp/<화자>.pt                       v0 패치
  runs/<RUN_NAME>/                                     복제본·채점 결과
  mask_exp/   surrogate_exp/   ctrl_exp/               v1, v2, 대조군 산출물
```

1. `01` → `02`(보호본) → `02`(대조군 칸으로 D4) → `03` 순서로 실행한다.
2. 개조 버전은 `variants/<버전>/`의 `01` → `02` → `03`을 차례로 실행한다. 버전별 `02`, `03`은 입출력 경로만 다르고, C1·C4는 보호 기법과 무관하므로 v0 결과를 복사해 재사용한다.

**재현할 때 주의**
- De-AntiFake의 전사 캐시(`TEXT_CACHE`)는 **실험마다 새 폴더**를 쓴다. 재사용하면 다른 실험의 전사가 섞인다 (실제로 한 번 오염돼 재실행함).
- De-AntiFake 16단계 정화가 끝난 뒤 15단계를 다시 돌리면 정화 결과가 지워진다. 백업 감시 칸이 20초마다 드라이브로 복사한다.
- conda 환경 생성(`02` 4단계) 중에 멈추면 패키지가 깨져 `CondaVerificationError`가 난다. `conda clean --all` 후 다시 만든다.

## 데이터 정책

이 저장소에는 **음성 파일을 올리지 않는다.** 노트북 출력(재생 위젯에 음성이 들어감)도 모두 지운 상태로 올렸다. LibriSpeech test-clean은 [OpenSLR](https://www.openslr.org/12)에서 받는다(CC BY 4.0).

## 사용한 코드와 모델

- Enkidu: [NoobCodeNameless/Enkidu](https://github.com/NoobCodeNameless/Enkidu) (ACM MM 2025). `v2`의 `apply()`는 Enkidu의 `add_noise`를 다시 구현한 것이다.
- De-AntiFake: [cyberrrange/De-AntiFake](https://github.com/cyberrrange/De-AntiFake) (ICML 2025)
- XTTS-v2: Coqui TTS 0.22.0 (모델은 비상업용 CPML 라이선스)
- 화자 인증: SpeechBrain `spkrec-ecapa-voxceleb`, Resemblyzer
- 심리음향 마스킹: MPEG-1 심리음향 모델 1, Qin et al. (ICML 2019)의 정식화
