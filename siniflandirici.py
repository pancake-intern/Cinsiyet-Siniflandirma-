"""
siniflandirici.py
─────────────────
Kural Tabanlı (Rule-Based) Cinsiyet Sınıflandırıcı

Literatür referans F0 aralıkları:
  Erkek : 85  – 155 Hz
  Kadın : 165 – 255 Hz
  Çocuk : 250 – 400 Hz

  Kaynak: Titze, I.R. (1994). Principles of Voice Production.
"""

# ──────────────────────────────────────────────
# SINIFLANDIRICI EŞİKLERİ
# (Veri setine göre ayarlayabilirsiniz)
# ──────────────────────────────────────────────
F0_ERKEK_UST   = 160   # F0 < 160 Hz  → Erkek
F0_COCUK_ALT   = 255   # F0 > 255 Hz  → Çocuk
ZCR_COCUK_ESIK = 0.08  # Kadın / Çocuk ayrımı için ZCR eşiği


def siniflandir(features: dict) -> str:
    """
    Özniteliklere göre kural tabanlı cinsiyet tahmini yapar.

    Kural sırası:
        1. F0 < F0_ERKEK_UST          → Erkek
        2. F0 > F0_COCUK_ALT          → Çocuk
        3. Aradaysa ZCR'e bak:
             ZCR > ZCR_COCUK_ESIK     → Çocuk
             ZCR ≤ ZCR_COCUK_ESIK     → Kadın

    Döndürür: 'Erkek' | 'Kadın' | 'Çocuk'
    """
    f0  = features['mean_f0']
    zcr = features['mean_zcr']

    if f0 < F0_ERKEK_UST:
        return 'Erkek'
    elif f0 > F0_COCUK_ALT:
        return 'Çocuk'
    else:
        # Geçiş bölgesi: ZCR ile ince ayar
        if zcr > ZCR_COCUK_ESIK:
            return 'Çocuk'
        else:
            return 'Kadın'
