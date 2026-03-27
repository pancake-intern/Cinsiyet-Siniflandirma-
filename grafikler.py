"""
grafikler.py
────────────
Proje için gerekli tüm grafikler:

  1. plot_acf_vs_fft()      → Otokorelasyon vs FFT (Bölüm 3A)
  2. plot_f0_dagilimi()     → Cinsiyet bazlı F0 kutu grafiği
  3. plot_confusion_matrix()→ Karışıklık matrisi
  4. plot_ste_zcr()         → STE ve ZCR zaman serisi (tek dosya)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from analiz import (
    load_audio, frame_signal, detect_voiced_frames,
    autocorrelation, estimate_f0_autocorrelation, estimate_f0_fft,
    SAMPLE_RATE
)


CIKTI_KLASOR = 'cikti'   # Grafiklerin kaydedileceği klasör


def _klasor_hazirla():
    os.makedirs(CIKTI_KLASOR, exist_ok=True)


# ──────────────────────────────────────────────
# 1. OTOKORELASYON vs FFT  (Bölüm 3A gereksinimi)
# ──────────────────────────────────────────────

def plot_acf_vs_fft(filepath: str, sr: int = SAMPLE_RATE):
    """
    Seçilen ses dosyası için:
        - Dalga formu
        - Otokorelasyon R(τ) + tespit edilen F0 işareti
        - FFT büyüklük spektrumu + tespit edilen F0 işareti
    yan yana çizer ve kaydeder.
    """
    _klasor_hazirla()

    audio, sr  = load_audio(filepath, sr=sr)
    frames, _, _ = frame_signal(audio, sr)
    voiced_mask, energies, _ = detect_voiced_frames(frames, sr)
    voiced_frames = frames[voiced_mask]

    if len(voiced_frames) == 0:
        print("[GRAFİK] Sesli çerçeve bulunamadı.")
        return None, None

    # Ortadaki sesli çerçeveyi örnek al
    frame = voiced_frames[len(voiced_frames) // 2]
    N     = len(frame)
    t_ms  = np.arange(N) / sr * 1000

    acf   = autocorrelation(frame)
    lags  = np.arange(len(acf)) / sr * 1000

    fft_mag = np.abs(np.fft.rfft(frame * np.hanning(N)))
    freqs   = np.fft.rfftfreq(N, d=1.0 / sr)
    fft_db  = 20 * np.log10(fft_mag + 1e-10)

    f0_acf = estimate_f0_autocorrelation(frame, sr)
    f0_fft = estimate_f0_fft(frame, sr)

    # ── Çizim ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(
        f'Otokorelasyon vs FFT — {os.path.basename(filepath)}',
        fontsize=13, fontweight='bold'
    )

    # Dalga formu
    axes[0].plot(t_ms, frame, color='steelblue', lw=0.8)
    axes[0].set_title('Zaman Düzlemi (Örnek Çerçeve)')
    axes[0].set_xlabel('Zaman (ms)')
    axes[0].set_ylabel('Genlik')
    axes[0].grid(True, alpha=0.3)

    # Otokorelasyon
    max_lag_ms = 1000 / 50   # 50 Hz alt sınırına kadar göster
    mask_lag   = lags <= max_lag_ms
    axes[1].plot(lags[mask_lag], acf[mask_lag], color='darkorange', lw=1.2)
    if f0_acf:
        lag_f0_ms = (sr / f0_acf) / sr * 1000
        axes[1].axvline(lag_f0_ms, color='red', ls='--', lw=1.8,
                        label=f'F₀ ≈ {f0_acf:.1f} Hz')
        axes[1].legend(fontsize=9)
    axes[1].set_title('Otokorelasyon  R(τ)')
    axes[1].set_xlabel('Gecikme τ (ms)')
    axes[1].set_ylabel('R(τ)')
    axes[1].grid(True, alpha=0.3)

    # FFT Spektrumu
    freq_mask = freqs <= 800
    axes[2].plot(freqs[freq_mask], fft_db[freq_mask],
                 color='mediumseagreen', lw=1.2)
    if f0_fft:
        axes[2].axvline(f0_fft, color='red', ls='--', lw=1.8,
                        label=f'F₀ ≈ {f0_fft:.1f} Hz')
        axes[2].legend(fontsize=9)
    axes[2].set_title('FFT Büyüklük Spektrumu')
    axes[2].set_xlabel('Frekans (Hz)')
    axes[2].set_ylabel('Genlik (dB)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    kayit = os.path.join(CIKTI_KLASOR, 'acf_vs_fft.png')
    plt.savefig(kayit, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[GRAFİK] {kayit} kaydedildi.')
    return f0_acf, f0_fft


# ──────────────────────────────────────────────
# 2. F0 DAĞILIMI  (Bölüm 5 gereksinimi)
# ──────────────────────────────────────────────

def plot_f0_dagilimi(df):
    """
    Gerçek etiketlere göre F0 kutu grafiği + nokta dağılımı.
    df: Ortalama_F0_Hz ve Gercek sütunları olan DataFrame
    """
    _klasor_hazirla()

    siniflar = ['Erkek', 'Kadın', 'Çocuk']
    renkler  = ['#4C72B0', '#DD8452', '#55A868']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Cinsiyet Bazlı F0 Dağılımı', fontsize=13, fontweight='bold')

    veri = [
        df[df['Gercek'] == s]['Ortalama_F0_Hz'].dropna().values
        for s in siniflar
    ]
    veri_mevcut = [v for v in veri if len(v) > 0]
    sinif_mevcut = [s for s, v in zip(siniflar, veri) if len(v) > 0]
    renk_mevcut  = [r for r, v in zip(renkler,  veri) if len(v) > 0]

    bp = axes[0].boxplot(veri_mevcut, labels=sinif_mevcut,
                         patch_artist=True, notch=False)
    for patch, renk in zip(bp['boxes'], renk_mevcut):
        patch.set_facecolor(renk)
        patch.set_alpha(0.7)
    axes[0].set_ylabel('Ortalama F0 (Hz)')
    axes[0].set_title('Kutu Grafiği')
    axes[0].grid(True, alpha=0.3)

    for sinif, renk in zip(siniflar, renkler):
        vals = df[df['Gercek'] == sinif]['Ortalama_F0_Hz'].dropna()
        if len(vals) > 0:
            axes[1].scatter([sinif] * len(vals), vals,
                            color=renk, alpha=0.65, s=70, label=sinif)
    axes[1].set_ylabel('Ortalama F0 (Hz)')
    axes[1].set_title('Nokta Dağılımı')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    kayit = os.path.join(CIKTI_KLASOR, 'f0_dagilimi.png')
    plt.savefig(kayit, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[GRAFİK] {kayit} kaydedildi.')


# ──────────────────────────────────────────────
# 3. KARISIKLIK MATRİSİ  (Bölüm 5 gereksinimi)
# ──────────────────────────────────────────────

def plot_confusion_matrix(df):
    """
    Karışıklık matrisini görselleştirir.
    df: Gercek ve Tahmin sütunları olan DataFrame
    """
    _klasor_hazirla()

    siniflar = ['Erkek', 'Kadın', 'Çocuk']
    n        = len(siniflar)
    cm       = np.zeros((n, n), dtype=int)

    etiketli = df.dropna(subset=['Gercek', 'Tahmin'])
    for _, row in etiketli.iterrows():
        if row['Gercek'] in siniflar and row['Tahmin'] in siniflar:
            i = siniflar.index(row['Gercek'])
            j = siniflar.index(row['Tahmin'])
            cm[i, j] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(siniflar, fontsize=11)
    ax.set_yticklabels(siniflar, fontsize=11)
    ax.set_xlabel('Tahmin Edilen Sınıf', fontsize=11)
    ax.set_ylabel('Gerçek Sınıf', fontsize=11)
    ax.set_title('Karışıklık Matrisi (Confusion Matrix)', fontsize=12, fontweight='bold')

    esik = cm.max() / 2
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center', fontsize=16, fontweight='bold',
                    color='white' if cm[i, j] > esik else 'black')

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    kayit = os.path.join(CIKTI_KLASOR, 'confusion_matrix.png')
    plt.savefig(kayit, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[GRAFİK] {kayit} kaydedildi.')
    return cm


# ──────────────────────────────────────────────
# 4. STE ve ZCR ZAMAN SERİSİ  (tek dosya)
# ──────────────────────────────────────────────

def plot_ste_zcr(filepath: str, sr: int = SAMPLE_RATE):
    """
    Tek ses dosyası için STE ve ZCR zaman serisini çizer.
    Sesli bölgeler vurgulanır.
    """
    _klasor_hazirla()

    audio, sr    = load_audio(filepath, sr=sr)
    frames, _, hop_len = frame_signal(audio, sr)
    voiced_mask, energies, zcrs = detect_voiced_frames(frames, sr)

    zaman = np.arange(len(frames)) * hop_len / sr  # saniye cinsinden

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f'STE / ZCR Analizi — {os.path.basename(filepath)}',
                 fontsize=12, fontweight='bold')

    # Dalga formu
    t = np.linspace(0, len(audio) / sr, len(audio))
    axes[0].plot(t, audio, color='steelblue', lw=0.5)
    axes[0].set_ylabel('Genlik')
    axes[0].set_title('Dalga Formu')
    axes[0].grid(True, alpha=0.3)

    # STE
    axes[1].plot(zaman, energies, color='darkorange', lw=1.2)
    axes[1].fill_between(zaman, energies, where=voiced_mask,
                         alpha=0.4, color='green', label='Sesli')
    axes[1].set_ylabel('STE')
    axes[1].set_title('Kısa Süreli Enerji (STE)')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # ZCR
    axes[2].plot(zaman, zcrs, color='mediumvioletred', lw=1.2)
    axes[2].fill_between(zaman, zcrs, where=voiced_mask,
                         alpha=0.4, color='green', label='Sesli')
    axes[2].set_ylabel('ZCR')
    axes[2].set_xlabel('Zaman (s)')
    axes[2].set_title('Sıfır Geçiş Oranı (ZCR)')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    isim  = os.path.splitext(os.path.basename(filepath))[0]
    kayit = os.path.join(CIKTI_KLASOR, f'ste_zcr_{isim}.png')
    plt.savefig(kayit, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[GRAFİK] {kayit} kaydedildi.')
