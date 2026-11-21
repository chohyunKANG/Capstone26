# v2 — XTTS-v2를 대리 모델로 (진행 중)

원본 Enkidu는 ECAPA 임베딩을 흔들도록 패치를 학습한다. v2는 실제 공격에 쓰는 XTTS-v2가 참조 음성에서 뽑는 두 값을 직접 흔든다.

| 값 | 코드 위치 (TTS 0.22.0) | 입력 |
|---|---|---|
| 화자 임베딩 (512) | `Xtts.hifigan_decoder.speaker_encoder` | 16 kHz |
| GPT 조건 잠재값 (1024×32) | `wav_to_mel_cloning` → `Xtts.gpt.get_style_emb` | 22.05 kHz 멜 |

원래 함수(`get_conditioning_latents`)는 `@torch.inference_mode()`라 미분이 안 되므로, 같은 처리를 grad가 흐르게 다시 짰다. 참조 길이와 청크는 모델 설정값을 읽어 실제 복제와 맞춘다.

나머지(패치 모양, 타일링, noise_level, mask_ratio, 옵티마이저, 분할)는 원본 Enkidu와 같다. 복제 모델을 알고 맞추는 **화이트박스 조건**이므로 결과는 "복제 모델을 알 때의 최선"으로 해석한다.

## 파일

- `xtts_ufp.py` — `mode`: `train`(화자별 패치 학습), `screen`(원본 패치와 크기별 비교), `save`(보호본 저장). `xtts` conda 환경에서 `conda run`으로 실행.
- `v2_xtts_surrogate.ipynb` — 환경 준비부터 저장까지.

출력: `xtts_exp/surrogate_exp/xtts/`
