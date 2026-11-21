# v1 — 청각 마스킹 기반 섭동 배치

원본 Enkidu는 학습한 패치에 `noise_level` 하나를 곱해 스펙트로그램 전체에 같은 세기로 더한다. 그래서 가려줄 소리가 없는 무음·자음 구간에서 잡음이 그대로 들린다.

v1은 패치를 더할 때 주파수 빈·프레임마다 상한을 둔다.

```
상한(f,t) = κ · √θ(f,t)      θ: 원본 음성의 심리음향 마스킹 임계값 (파워)
넘는 빈만 상한까지 줄이고 위상은 유지. κ = ∞ 이면 원본 Enkidu와 같다.
```

κ=64는 마스킹 임계값보다 진폭 64배(약 36 dB) 위까지 허용하므로 **"들리지 않는 섭동"이 아니다.** κ=1이 이론상 가청 경계인데, 그때는 보호가 사라졌다.

## 파일

- `psychomask.py` — 마스킹 임계값(`masking_threshold`)과 사후 적용(`apply_masked`). numpy만 사용. 노트북이 드라이브에 그대로 써서 import한다.
- `01_protect_v1.ipynb` — 보호: 사후 마스킹(κ=1, κ 스윕), 균일 축소 대조군, Wiener 분해, 학습 단계 마스킹(`MaskedEnkidu`), 그림
- `02_purify_v1.ipynb` — 정화: κ=64 보호본 → `mask_exp/train_k64/D3`
- `03_clone_v1.ipynb` — 복제·채점: `runs/enkidu_mask_k64/` (C1·C4는 v0 것 재사용)

결과는 [`results/README.md`](../../results/README.md).
