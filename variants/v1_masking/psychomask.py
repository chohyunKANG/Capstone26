"""
Psychoacoustic masking threshold (MPEG-1 Psychoacoustic Model 1 style).
Reference formulation: Qin et al., "Imperceptible, Robust, and Targeted
Adversarial Examples for ASR", ICML 2019.

theta(f, t) = the power, per frequency bin per frame, that can be added
to x without being audible.  Depends ONLY on x, so it can be precomputed
once per utterance and treated as a constant inside the training loop.
"""

import numpy as np


# ---------------------------------------------------------------- helpers
def hz_to_bark(f):
    f = np.asarray(f, dtype=np.float64)
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def absolute_threshold_hearing(f):
    """ATH in dB SPL. f in Hz. Clamped below 20 Hz to avoid blow-up."""
    f = np.maximum(np.asarray(f, dtype=np.float64), 20.0)
    k = f / 1000.0
    return (3.64 * k ** (-0.8)
            - 6.5 * np.exp(-0.6 * (k - 3.3) ** 2)
            + 1e-3 * k ** 4)


def _spreading(dz, p_masker_db):
    """
    Two-slope spreading function, dB.
    dz = bark(maskee) - bark(masker).  Lower slope +27 dB/bark,
    upper slope level-dependent.
    """
    upper = -27.0 + 0.37 * np.maximum(p_masker_db - 40.0, 0.0)
    return np.where(dz < 0, 27.0 * dz, upper * dz)


# ---------------------------------------------------------- main routine
def masking_threshold(
    x,
    sr=16000,
    n_fft=1024,
    hop=256,
    win=None,
    max_maskers=48,
    center=True,
):
    """
    Parameters
    ----------
    x      : 1-D float array, waveform in [-1, 1]
    center : pad n_fft//2 on both sides, so the frame count matches
             torch.stft(..., center=True).  Keep True when the result is
             multiplied against a torch STFT.
    Returns
    -------
    theta : (n_bins, n_frames) float64, masking threshold in POWER units,
            on the same scale as |STFT(x)|**2 after the 96 dB normalisation
            is undone -- i.e. directly comparable to |STFT(x)|**2.
    """
    if win is None:
        win = np.hanning(n_fft)

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if center:
        x = np.pad(x, (n_fft // 2, n_fft // 2), mode="reflect")

    # ---- STFT ------------------------------------------------------
    n = len(x)
    n_frames = 1 + max(0, (n - n_fft) // hop)
    if n_frames < 1:
        x = np.pad(x, (0, n_fft - n))
        n_frames = 1
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * win[None, :]
    X = np.fft.rfft(frames, n=n_fft, axis=1)          # (T, K)
    mag2 = np.abs(X) ** 2
    n_bins = X.shape[1]

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    barks = hz_to_bark(freqs)
    ath = absolute_threshold_hearing(freqs)            # dB SPL

    # ---- PSD normalised so that the global max is 96 dB SPL --------
    eps = 1e-20
    psd_raw = 10.0 * np.log10(mag2 + eps)              # dB, arbitrary ref
    offset = 96.0 - psd_raw.max()
    psd = psd_raw + offset                             # dB SPL

    theta_db = np.empty((n_frames, n_bins), dtype=np.float64)

    for t in range(n_frames):
        p = psd[t]

        # ---- tonal maskers: local peaks standing >=7 dB above
        #      their neighbourhood -------------------------------------
        is_peak = np.zeros(n_bins, dtype=bool)
        is_peak[1:-1] = (p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:])
        cand = np.where(is_peak)[0]

        keep = []
        for k in cand:
            if k < 2 or k > n_bins - 3:
                continue
            # neighbourhood width grows with frequency (MPEG dk rule)
            if k < 63:
                dks = (2,)
            elif k < 127:
                dks = (2, 3)
            else:
                dks = (2, 3, 4, 5, 6)
            nb = []
            for dk in dks:
                if 0 <= k - dk and k + dk < n_bins:
                    nb.extend((p[k - dk], p[k + dk]))
            if nb and p[k] >= max(nb) + 7.0:
                keep.append(k)

        if len(keep) == 0:
            theta_db[t] = ath
            continue

        # strongest maskers only -- keeps the O(K*M) loop bounded
        keep = np.array(keep)
        if len(keep) > max_maskers:
            keep = keep[np.argsort(p[keep])[-max_maskers:]]

        # tonal masker level = sum of the peak and its two neighbours
        p_mask = 10.0 * np.log10(
            np.sum(10.0 ** (p[np.stack([keep - 1, keep, keep + 1])] / 10.0),
                   axis=0) + eps
        )                                              # (M,)

        # drop maskers already below the hearing threshold
        alive = p_mask >= ath[keep]
        if not alive.any():
            theta_db[t] = ath
            continue
        keep, p_mask = keep[alive], p_mask[alive]

        # ---- individual masking thresholds ------------------------
        dz = barks[None, :] - barks[keep][:, None]     # (M, K)
        T = (p_mask[:, None]
             - 0.275 * barks[keep][:, None]
             + _spreading(dz, p_mask[:, None])
             - 6.025)

        # only within +-3 bark does a masker contribute
        T = np.where(np.abs(dz) <= 3.0, T, -np.inf)

        # ---- global threshold: power-sum of maskers and ATH --------
        with np.errstate(over="ignore"):
            tot = np.sum(10.0 ** (T / 10.0), axis=0) + 10.0 ** (ath / 10.0)
        theta_db[t] = 10.0 * np.log10(tot + eps)

    # ---- back to the power scale of |STFT(x)|**2 ------------------
    theta = 10.0 ** ((theta_db - offset) / 10.0)
    return theta.T                                     # (K, T)


# ------------------------------------------------------- apply the gain
def shape_perturbation(delta_tiled, theta, kappa=1.0, noise_level=1.0):
    """
    Clip the perturbation to the masking ceiling, per bin.

        delta_out = noise_level * delta * min(1, kappa*sqrt(theta) / |noise_level*delta|)

    The patch keeps its learned magnitude wherever that already sits below
    what the ear can hide; it is pulled down only where it would stick out.
    Phase is never touched.

    delta_tiled : (K, T) complex, the tiled UFP
    theta       : (K, T) real, from masking_threshold()
    noise_level : Enkidu's existing global scale -- keep it
    kappa       : the new knob.  kappa -> inf reproduces plain Enkidu.

    Returns (delta_out, clip_fraction)
    """
    d = noise_level * delta_tiled
    mag = np.abs(d)
    ceil = kappa * np.sqrt(theta)
    gain = np.minimum(1.0, ceil / np.maximum(mag, 1e-12))
    clipped = float((gain < 0.999).mean())
    return d * gain, clipped


# ------------------------------------------- numpy STFT / iSTFT (matched)
def _stft(x, n_fft=1024, hop=256):
    x = np.pad(np.asarray(x, np.float64).reshape(-1),
               (n_fft // 2, n_fft // 2), mode="reflect")
    win = np.hanning(n_fft)
    nf = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(nf)[:, None]
    return np.fft.rfft(x[idx] * win[None, :], axis=1).T     # (K, T)


def _istft(S, length, n_fft=1024, hop=256):
    win = np.hanning(n_fft)
    frames = np.fft.irfft(S.T, n=n_fft, axis=1) * win[None, :]
    nf = frames.shape[0]
    total = n_fft + hop * (nf - 1)
    y = np.zeros(total); wsum = np.zeros(total)
    for t in range(nf):
        s = t * hop
        y[s:s + n_fft] += frames[t]
        wsum[s:s + n_fft] += win ** 2
    y /= np.maximum(wsum, 1e-8)
    y = y[n_fft // 2: n_fft // 2 + length]
    if len(y) < length:
        y = np.pad(y, (0, length - len(y)))
    return y


def apply_masked(x, y, kappa=1.0, sr=16000, n_fft=1024, hop=256):
    """
    x : clean waveform (1-D numpy)
    y : Enkidu output for the same x (1-D numpy), i.e. x + delta
    Returns (y_masked, info)

    Because STFT is linear, delta = y - x exactly.  delta is re-analysed,
    clipped bin-by-bin to kappa*sqrt(theta(x)), and added back.
    kappa = inf returns y unchanged.
    """
    x = np.asarray(x, np.float64).reshape(-1)
    y = np.asarray(y, np.float64).reshape(-1)
    n = min(len(x), len(y)); x, y = x[:n], y[:n]
    d = y - x
    if not np.isfinite(kappa):
        return y, {"clip_frac": 0.0, "energy_kept": 1.0}

    theta = masking_threshold(x, sr=sr, n_fft=n_fft, hop=hop, center=True)
    D = _stft(d, n_fft, hop)
    T = min(D.shape[1], theta.shape[1])
    D, theta = D[:, :T], theta[:, :T]
    mag = np.abs(D)
    gain = np.minimum(1.0, kappa * np.sqrt(theta) / np.maximum(mag, 1e-12))
    d2 = _istft(D * gain, n, n_fft, hop)
    info = {"clip_frac": float((gain < 0.999).mean()),
            "energy_kept": float((d2 ** 2).sum() / max((d ** 2).sum(), 1e-20))}
    return x + d2, info
