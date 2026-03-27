"""
analiz.py
─────────
Zaman Düzlemi Ses Analizi Modülü

İçerik:
  - Pencereleme (25 ms)
  - Kısa Süreli Enerji (STE)
  - Sıfır Geçiş Oranı (ZCR)
  - Sesli (Voiced) bölge tespiti
  - Otokorelasyon ile F0 tespiti  →  R(τ) = Σ x[n]·x[n−τ]
  - FFT ile F0 tespiti            →  kıyaslama için
  - Öznitelik çıkarımı (tek dosya)
"""

import numpy as np
import librosa
from scipy.signal import find_peaks


# ──────────────────────────────────────────────
# SABITLER
# ──────────────────────────────────────────────
SAMPLE_RATE  = 16000   # Hedef örnekleme hızı (Hz)
FRAME_MS     = 25      # Çerçeve uzunluğu (ms)  — talimatname: 20-30 ms
HOP_MS       = 10      # Atlama uzunluğu (ms)
F0_MIN_HZ    = 50      # Arama alt sınırı
F0_MAX_HZ    = 500     # Arama üst sınırı
STE_RATIO    = 0.02    # Sesli bölge enerji eşiği (max enerjinin %2'si)
ZCR_MAX      = 0.15    # Sesli bölge ZCR üst sınırı
ACF_MIN_PEAK = 0.30    # Otokorelasyon tepe eşiği


# ──────────────────────────────────────────────
# 1. SES YÜKLEME
# ──────────────────────────────────────────────

def load_audio(filepath: str, sr: int = SAMPLE_RATE):
    """
    Ses dosyasını yükler.
      - Mono'ya dönüştürür
      - DC offset kaldırır
      - [-1, 1] aralığına normalize eder
    """
    audio, sr_out = librosa.load(filepath, sr=sr, mono=True)
    audio = audio - np.mean(audio)                          # DC offset
    peak  = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak                                # normalize
    return audio, sr_out


# ──────────────────────────────────────────────
# 2. ÇERÇEVELEME
# ──────────────────────────────────────────────

def frame_signal(audio: np.ndarray, sr: int,
                 frame_ms: int = FRAME_MS, hop_ms: int = HOP_MS):
    """
    Sinyali örtüşen çerçevelere böler.
    Döndürür: (frames, frame_len, hop_len)
    """
    frame_len = int(sr * frame_ms / 1000)
    hop_len   = int(sr * hop_ms   / 1000)
    frames    = librosa.util.frame(
        audio, frame_length=frame_len, hop_length=hop_len
    ).T                                                     # (N_frames, frame_len)
    return frames, frame_len, hop_len


# ──────────────────────────────────────────────
# 3. ZAMAN DÜZLEMİ ÖZNİTELİKLERİ
# ──────────────────────────────────────────────

def compute_ste(frame: np.ndarray) -> float:
    """Kısa Süreli Enerji: E = (1/N) Σ x[n]²"""
    return float(np.sum(frame ** 2) / len(frame))


def compute_zcr(frame: np.ndarray) -> float:
    """
    Sıfır Geçiş Oranı: ZCR = (1/N) Σ |sgn(x[n]) - sgn(x[n-1])| / 2
    """
    signs = np.sign(frame)
    signs[signs == 0] = 1
    crossings = np.sum(np.abs(np.diff(signs))) / 2
    return float(crossings / len(frame))


def detect_voiced_frames(frames: np.ndarray, sr: int):
    """
    Her çerçeve için STE ve ZCR hesaplar.
    Sesli (voiced) çerçeveler:
      - STE > STE_RATIO × max(STE)   → yeterli enerji
      - ZCR < ZCR_MAX                → periyodik sinyal (düşük ZCR)

    Döndürür: voiced_mask, energies, zcrs
    """
    energies = np.array([compute_ste(f) for f in frames])
    zcrs     = np.array([compute_zcr(f) for f in frames])

    ste_threshold = STE_RATIO * np.max(energies)
    voiced_mask   = (energies > ste_threshold) & (zcrs < ZCR_MAX)

    return voiced_mask, energies, zcrs


# ──────────────────────────────────────────────
# 4. OTOKORELASYON ile F0 TESPİTİ
# ──────────────────────────────────────────────

def autocorrelation(frame: np.ndarray) -> np.ndarray:
    """
    Normalize otokorelasyon hesaplar:
        R(τ) = Σ x[n] · x[n−τ]    τ = 0, 1, 2, ...

    R(0) = 1 olacak şekilde normalize edilir.
    """
    n   = len(frame)
    acf = np.correlate(frame, frame, mode='full')
    acf = acf[n - 1:]                               # sadece τ ≥ 0
    acf = acf / (acf[0] + 1e-10)                   # R(0) = 1
    return acf


def estimate_f0_autocorrelation(frame: np.ndarray, sr: int) -> float | None:
    """
    Otokorelasyon tepe noktasından F0 tahmin eder.

    Geçerli gecikme (lag) aralığı:
        lag_min = sr / F0_MAX_HZ
        lag_max = sr / F0_MIN_HZ
    """
    acf     = autocorrelation(frame)
    lag_min = int(sr / F0_MAX_HZ)
    lag_max = int(sr / F0_MIN_HZ)

    if lag_max >= len(acf):
        return None

    search       = acf[lag_min:lag_max]
    peaks, props = find_peaks(search, height=ACF_MIN_PEAK)

    if len(peaks) == 0:
        return None

    best_peak = peaks[np.argmax(props['peak_heights'])]
    best_lag  = best_peak + lag_min
    f0        = sr / best_lag
    return float(f0)


# ──────────────────────────────────────────────
# 5. FFT ile F0 TESPİTİ  (kıyaslama)
# ──────────────────────────────────────────────

def estimate_f0_fft(frame: np.ndarray, sr: int) -> float | None:
    """
    FFT büyüklük spektrumundan F0 tahmin eder.
    Sadece Otokorelasyon ile kıyaslama grafiği için kullanılır.
    """
    N      = len(frame)
    window = np.hanning(N)
    fft    = np.abs(np.fft.rfft(frame * window))
    freqs  = np.fft.rfftfreq(N, d=1.0 / sr)

    # Sadece F0_MIN – F0_MAX aralığını ara
    mask       = (freqs >= F0_MIN_HZ) & (freqs <= F0_MAX_HZ)
    fft_masked = fft.copy()
    fft_masked[~mask] = 0

    peaks, _ = find_peaks(fft_masked, height=np.max(fft_masked) * 0.3)
    if len(peaks) == 0:
        return None

    return float(freqs[peaks[0]])


# ──────────────────────────────────────────────
# 6. TEK DOSYA ÖZNİTELİK ÇIKARIMI
# ──────────────────────────────────────────────

def extract_features(filepath: str) -> dict | None:
    """
    Bir ses dosyasından öznitelikleri çıkarır:
        - mean_f0   : Otokorelasyon ile ortalama F0 (Hz)
        - std_f0    : F0 standart sapması
        - mean_zcr  : Sesli bölge ortalama ZCR
        - mean_ste  : Sesli bölge ortalama STE
        - n_voiced  : Sesli çerçeve sayısı
        - f0_values : Tüm çerçeve F0 listesi (grafik için)

    Başarısız olursa None döndürür.
    """
    try:
        audio, sr = load_audio(filepath)
    except Exception as e:
        print(f"    [HATA] {filepath} yüklenemedi: {e}")
        return None

    frames, _, _ = frame_signal(audio, sr)

    if len(frames) == 0:
        return None

    voiced_mask, energies, zcrs = detect_voiced_frames(frames, sr)
    voiced_frames = frames[voiced_mask]

    if len(voiced_frames) < 3:
        return None

    # Her sesli çerçeve için otokorelasyon ile F0 hesapla
    f0_values = []
    for f in voiced_frames:
        f0 = estimate_f0_autocorrelation(f, sr)
        if f0 is not None:
            f0_values.append(f0)

    if len(f0_values) == 0:
        return None

    return {
        'mean_f0':   float(np.mean(f0_values)),
        'std_f0':    float(np.std(f0_values)),
        'mean_zcr':  float(np.mean(zcrs[voiced_mask])),
        'mean_ste':  float(np.mean(energies[voiced_mask])),
        'n_voiced':  int(np.sum(voiced_mask)),
        'f0_values': f0_values,
    }
