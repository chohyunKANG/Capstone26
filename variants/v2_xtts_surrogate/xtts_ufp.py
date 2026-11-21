"""
Enkidu-style UFP trained against XTTS-v2's own conditioning (candidate 2).

Same patch shape / tiling / noise_level / masking / optimiser as Enkidu.
Only the surrogate changes: ECAPA  ->  the two things XTTS-v2 extracts
from a reference clip (TTS 0.22.0, Xtts.get_conditioning_latents):
  * speaker_embedding : hifigan_decoder.speaker_encoder, 16 kHz, 512-d
  * gpt_cond_latent   : wav_to_mel_cloning -> gpt.get_style_emb, 22.05 kHz

usage:  python xtts_ufp.py cfg.json      (cfg["mode"] = train | screen | save)
"""
import os, sys, json, time, random
from pathlib import Path
import numpy as np, torch, torchaudio, soundfile as sf
import torch.nn.functional as F

cfg = json.load(open(sys.argv[1]))
MODE, DEV, SR = cfg["mode"], "cuda", 16000

from TTS.api import TTS
from TTS.tts.models.xtts import wav_to_mel_cloning

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
M, C = tts.synthesizer.tts_model, tts.synthesizer.tts_config
M.eval()
for p in M.parameters():
    p.requires_grad_(False)
GPT_LEN, GPT_CHUNK, MAX_REF = C.gpt_cond_len, C.gpt_cond_chunk_len, C.max_ref_len
PERC = M.args.gpt_use_perceiver_resampler
print(f"XTTS 참조 처리: max_ref_len={MAX_REF}s, gpt_cond_len={GPT_LEN}s, "
      f"chunk={GPT_CHUNK}s, perceiver={PERC}, sound_norm={C.sound_norm_refs}", flush=True)


# ------------------------------------------------ XTTS conditioning (with grad)
def cond(x16):
    """x16: (1, N) 16 kHz on DEV -> (speaker_emb, gpt_latent), both flat."""
    a = torchaudio.functional.resample(x16, SR, 22050)[:, : 22050 * MAX_REF]
    if C.sound_norm_refs:
        a = a / a.abs().max() * 0.75
    a16 = torchaudio.functional.resample(a, 22050, 16000)
    spk = M.hifigan_decoder.speaker_encoder.forward(a16.clone(), l2_norm=True).flatten()

    g = a[:, : 22050 * GPT_LEN]
    if PERC:
        embs, step = [], 22050 * GPT_CHUNK
        for i in range(0, g.shape[1], step):
            ch = g[:, i:i + step]
            if ch.size(-1) < 22050 * 0.33:
                continue
            mel = wav_to_mel_cloning(ch, mel_norms=M.mel_stats, device=ch.device,
                                     n_fft=2048, hop_length=256, win_length=1024,
                                     power=2, normalized=False, sample_rate=22050,
                                     f_min=0, f_max=8000, n_mels=80)
            embs.append(M.gpt.get_style_emb(mel, None))
        gpt = torch.stack(embs).mean(0)
    else:
        mel = wav_to_mel_cloning(g, mel_norms=M.mel_stats, device=g.device,
                                 n_fft=4096, hop_length=1024, win_length=4096,
                                 power=2, normalized=False, sample_rate=22050,
                                 f_min=0, f_max=8000, n_mels=80)
        gpt = M.gpt.get_style_emb(mel)
    return spk, gpt.flatten()


# ------------------------------------------ Enkidu's add_noise (Wiener off)
NF, HOP, WIN, L = 1024, 512, 1024, cfg["frame_length"]
win = torch.hann_window(NF, device=DEV)

def apply(x16, nr, ni, level, mask_ratio=0.0):
    x = x16.flatten()
    S = torch.stft(x, NF, HOP, WIN, window=win, return_complex=True).unsqueeze(0)
    T = S.shape[-1]
    Dr, Di = torch.zeros_like(S.real), torch.zeros_like(S.imag)
    if L > T:
        Dr[:, :, :T] = nr[:, :, :T]; Di[:, :, :T] = ni[:, :, :T]
    else:
        n = T // L
        keep = torch.rand(n) >= mask_ratio
        if mask_ratio > 0 and keep.sum() == 0:
            keep[random.randrange(n)] = True
        for i in range(n):
            if keep[i]:
                Dr[:, :, i * L:(i + 1) * L] = nr
                Di[:, :, i * L:(i + 1) * L] = ni
    S2 = torch.complex(S.real + level * Dr, S.imag + level * Di)
    y = torch.istft(S2, NF, HOP, WIN, window=win, length=x.numel())
    return y.clamp(-1.0, 1.0).reshape(1, -1)


# ----------------------------------------------------------------- data
def load16(p):
    y, sr = sf.read(p, dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    w = torch.from_numpy(y).unsqueeze(0)
    if sr != SR:
        w = torchaudio.functional.resample(w, sr, SR)
    return w.to(DEV)

D1, EVAL, OUT = Path(cfg["d1"]), Path(cfg["eval_ref"]), Path(cfg["out"])
(OUT / "ufp").mkdir(parents=True, exist_ok=True)
SPKS = sorted(d.name for d in D1.iterdir() if d.is_dir() and not d.name.startswith("."))

def split(spk):
    """Same split as the Enkidu notebook: sorted, first 70 % train."""
    paths = sorted(str(p) for p in (D1 / spk).glob("*.wav"))
    k = max(1, int(len(paths) * 0.7))
    tr, ev = paths[:k], paths[k:]
    have = {p.name for p in (EVAL / spk).glob("*.wav")}
    assert {Path(p).name for p in ev} == have, \
        f"{spk}: 평가셋이 기존 D2와 다름 (분할 {len(ev)}개 / D2 {len(have)}개)"
    return tr, ev


# ---------------------------------------------------------------- train
def train(spk):
    pt = OUT / "ufp" / f"{spk}.pt"
    if pt.exists():
        print(f"[{spk}] 기존 패치 사용", flush=True); return
    tr, _ = split(spk)
    xs = [load16(p) for p in tr]
    with torch.no_grad():
        refs = [cond(x) for x in xs]
    seed = cfg["seed"] + SPKS.index(spk)
    torch.manual_seed(seed); random.seed(seed)
    nr = torch.randn((1, NF // 2 + 1, L), device=DEV, requires_grad=True)
    ni = torch.randn((1, NF // 2 + 1, L), device=DEV, requires_grad=True)
    opt = torch.optim.Adam([nr, ni], lr=cfg["alpha"], weight_decay=cfg["decay"])
    t0 = time.time()
    for ep in range(cfg["steps"]):
        opt.zero_grad(); s_sum = g_sum = 0.0
        for x, (s0, g0) in zip(xs, refs):
            s1, g1 = cond(apply(x, nr, ni, cfg["noise_level"], cfg["mask_ratio"]))
            cs, cg = F.cosine_similarity(s0, s1, dim=0), F.cosine_similarity(g0, g1, dim=0)
            (cs + cg).backward()
            s_sum += cs.item(); g_sum += cg.item()
        opt.step()
        if ep == 0 or (ep + 1) % 10 == 0:
            print(f"  [{spk}] epoch {ep+1:>3}/{cfg['steps']}  "
                  f"speaker cos {s_sum/len(xs):.3f}  gpt cos {g_sum/len(xs):.3f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
    torch.save({"real": nr.detach().cpu(), "imag": ni.detach().cpu(),
                "surrogate": "xtts_v2", **{k: cfg[k] for k in
                ("noise_level", "frame_length", "mask_ratio", "steps", "alpha", "decay")}}, pt)
    print(f"[{spk}] 학습 {len(xs)}개 · {(time.time()-t0)/60:.1f}분 · 저장", flush=True)


# ---------------------------------------------------- screen / save
def patches(src):
    out = {}
    for spk in SPKS:
        d = torch.load(Path(src) / f"{spk}.pt")
        out[spk] = (d["real"].to(DEV), d["imag"].to(DEV))
    return out

def screen():
    from pystoi import stoi
    srcs = {"ECAPA (원본 Enkidu)": cfg["baseline_ufp"], "XTTS (후보 2)": str(OUT / "ufp")}
    rows = []
    for name, src in srcs.items():
        P = patches(src)
        for spk in SPKS:
            _, ev = split(spk)
            nr, ni = P[spk]
            for p in ev:
                x = load16(p)
                with torch.no_grad():
                    s0, g0 = cond(x)
                    xn = x[0].cpu().numpy()
                    for c in cfg["scales"]:
                        y = apply(x, nr, ni, cfg["noise_level"] * c)
                        s1, g1 = cond(y)
                        rows.append((name, c,
                                     F.cosine_similarity(s0, s1, dim=0).item(),
                                     F.cosine_similarity(g0, g1, dim=0).item(),
                                     stoi(xn, y[0].cpu().numpy(), SR, extended=False)))
        print(f"{name} 완료", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows, columns=["surrogate", "scale", "xtts_spk_cos", "xtts_gpt_cos", "stoi"])
    df.to_csv(OUT / "screen.csv", index=False)
    print(df.groupby(["surrogate", "scale"]).mean().round(3).sort_index(ascending=[True, False]).to_string())

def save():
    c = cfg["save_scale"]
    P = patches(OUT / "ufp")
    dst = OUT / f"D2_c{c:g}"
    n = 0
    for spk in SPKS:
        _, ev = split(spk)
        nr, ni = P[spk]
        (dst / spk).mkdir(parents=True, exist_ok=True)
        for p in ev:
            with torch.no_grad():
                y = apply(load16(p), nr, ni, cfg["noise_level"] * c)
            sf.write(str(dst / spk / Path(p).name), y[0].cpu().numpy(), SR, subtype="FLOAT")
            n += 1
    print(f"저장 {n}개 -> {dst}", flush=True)


if MODE == "train":
    for spk in SPKS:
        train(spk)
elif MODE == "screen":
    screen()
elif MODE == "save":
    save()
