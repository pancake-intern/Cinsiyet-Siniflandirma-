"""
main.py
───────
Ses İşareti Analizi ve Cinsiyet Sınıflandırma — Ana Program

Kullanım:
    python main.py                  →  tüm Dataset/ analizi
    python main.py dosya.wav        →  tek dosya tahmini

Beklenen klasör yapısı:
    proje/
    ├── main.py
    ├── analiz.py
    ├── siniflandirici.py
    ├── grafikler.py
    ├── requirements.txt
    └── Dataset/
            metadata.xlsx           ← TEK birleşik Excel dosyası
            GRUP_05/
                G05_D01_C_9_Neutral_C1.wav
                ...
            GROUP_01/
                G01_D01_E_25_Happy_C1.wav
                ...

metadata.xlsx sütunları (group5.xlsx formatına göre):
    FILE NAME | Subject_ID | Gender | Age | Feeling | ...
"""

import os
import sys
import glob
import pandas as pd
import numpy as np

from analiz          import extract_features, load_audio, frame_signal, detect_voiced_frames
from siniflandirici  import siniflandir
from grafikler       import (
    plot_acf_vs_fft, plot_f0_dagilimi,
    plot_confusion_matrix, plot_ste_zcr, CIKTI_KLASOR
)

DATASET_KLASOR = 'Dataset'
GENDER_MAP     = {'E': 'Erkek', 'K': 'Kadın', 'C': 'Çocuk', 'Ç': 'Çocuk'}


# ──────────────────────────────────────────────
# VERİ OKUMA  (Talimatname Bölüm 2)
# ──────────────────────────────────────────────

def veri_oku(dataset_klasor: str = DATASET_KLASOR) -> pd.DataFrame:
    """
    Dataset/ klasöründeki TEK metadata.xlsx dosyasını okur.

    Talimatname adımları:
        1. pandas.read_excel() ile metadata dosyasını oku
        2. Her satır için dosya yolunu oluştur
        3. os.path.exists() ile dosyanın varlığını kontrol et
        4. Döngüyle tüm wav dosyalarını işleme al
    """

    # ── Metadata Excel dosyasını bul ──────────────────────
    excel_yolu = os.path.join(dataset_klasor, 'metadata.xlsx')

    if not os.path.exists(excel_yolu):
        # Alternatif: herhangi bir xlsx ara
        excel_listesi = glob.glob(
            os.path.join(dataset_klasor, '*.xlsx'), recursive=False
        )
        if not excel_listesi:
            print(f'[HATA] {dataset_klasor}/metadata.xlsx bulunamadı.')
            print('       Lütfen birleşik metadata.xlsx dosyasını Dataset/ altına koyun.')
            sys.exit(1)
        excel_yolu = excel_listesi[0]
        print(f'[BİLGİ] Excel dosyası: {excel_yolu}')

    # ── pandas.read_excel() ile oku ───────────────────────
    try:
        df = pd.read_excel(excel_yolu)
    except Exception as e:
        print(f'[HATA] Excel okunamadı: {e}')
        sys.exit(1)

    # Sütun adlarını normalize et
    df.columns = df.columns.str.strip()
    print(f'[VERİ] {len(df)} satır okundu.')
    print(f'[VERİ] Sütunlar: {list(df.columns)}')

    # ── Dosya yollarını oluştur ve kontrol et ─────────────
    kayitlar = []

    for _, satir in df.iterrows():
        dosya_adi = str(satir.get('FILE NAME', '')).strip()
        if not dosya_adi:
            continue
        if not dosya_adi.lower().endswith('.wav'):
            dosya_adi += '.wav'

        # Grup klasörü adını dosya adından çıkar
        # Örnek: G05_D01_C_9_Neutral_C1.wav → GRUP_05 veya GROUP_05 veya GRUP05
        grup_kodu = dosya_adi.split('_')[0]                 # G05
        grup_no   = grup_kodu.replace('G', '').lstrip('0')  # 5
        grup_no_2 = grup_kodu.replace('G', '').zfill(2)     # 05

        # Olası klasör isimleri
        adaylar = [
            os.path.join(dataset_klasor, f'GRUP_{grup_no_2}', dosya_adi),
            os.path.join(dataset_klasor, f'GROUP_{grup_no_2}', dosya_adi),
            os.path.join(dataset_klasor, f'Grup_{grup_no_2}', dosya_adi),
            os.path.join(dataset_klasor, f'Group_{grup_no_2}', dosya_adi),
            os.path.join(dataset_klasor, f'GRUP_{grup_no}',   dosya_adi),
            os.path.join(dataset_klasor, f'GROUP_{grup_no}',  dosya_adi),
            os.path.join(dataset_klasor, dosya_adi),           # düz kök
        ]

        # os.path.exists() ile varlık kontrolü (talimatname gereği)
        dosya_yolu = next((a for a in adaylar if os.path.exists(a)), None)

        if dosya_yolu is None:
            print(f'  [YOK] {dosya_adi}')
            continue

        gender_raw = str(satir.get('Gender', '')).strip().upper()
        cinsiyet   = GENDER_MAP.get(gender_raw)

        try:
            yas = int(satir.get('Age', 0))
        except (ValueError, TypeError):
            yas = None

        kayitlar.append({
            'Dosya_Yolu': dosya_yolu,
            'Dosya_Adi':  dosya_adi,
            'Cinsiyet':   cinsiyet,
            'Yas':        yas,
            'Duygu':      str(satir.get('Feeling', '')).strip(),
        })

    if not kayitlar:
        print('[HATA] Hiç geçerli ses dosyası bulunamadı.')
        sys.exit(1)

    master = pd.DataFrame(kayitlar)
    print(f'\n[VERİ] Geçerli kayıt sayısı: {len(master)}')
    print(f'[VERİ] Cinsiyet dağılımı:\n{master["Cinsiyet"].value_counts().to_string()}\n')
    return master


# ──────────────────────────────────────────────
# ANALİZ DÖNGÜSÜ  (Talimatname Bölüm 3)
# ──────────────────────────────────────────────

def analiz_pipeline(master: pd.DataFrame) -> pd.DataFrame:
    """
    Tüm wav dosyaları üzerinde döngüyle:
        1. Öznitelik çıkar (F0, ZCR, STE)
        2. Kural tabanlı sınıflandır
        3. Sonuçları topla
    """
    sonuclar = []
    toplam   = len(master)

    print('─' * 60)
    print(f'{"Dosya":<35} {"Gerçek":<8} {"Tahmin":<8} {"F0 (Hz)":<10} {"OK"}')
    print('─' * 60)

    for idx, satir in master.iterrows():
        yol      = satir['Dosya_Yolu']
        gercek   = satir['Cinsiyet']
        duygu    = satir.get('Duygu', '')

        ozellikler = extract_features(yol)

        if ozellikler is None:
            print(f'  [ATLA] {os.path.basename(yol)} — öznitelik çıkarılamadı')
            continue

        tahmin  = siniflandir(ozellikler)
        dogru   = (tahmin == gercek) if gercek else None
        simge   = '✓' if dogru else ('✗' if dogru is False else '?')

        print(f'{os.path.basename(yol):<35} {str(gercek):<8} {tahmin:<8} '
              f'{ozellikler["mean_f0"]:<10.1f} {simge}')

        sonuclar.append({
            'Dosya_Adi':      os.path.basename(yol),
            'Dosya_Yolu':     yol,
            'Gercek':         gercek,
            'Tahmin':         tahmin,
            'Dogru':          dogru,
            'Duygu':          duygu,
            'Ortalama_F0_Hz': round(ozellikler['mean_f0'], 2),
            'Std_F0_Hz':      round(ozellikler['std_f0'],  2),
            'Ortalama_ZCR':   round(ozellikler['mean_zcr'], 4),
            'Ortalama_STE':   round(ozellikler['mean_ste'], 6),
            'Sesli_Cerceve':  ozellikler['n_voiced'],
        })

    return pd.DataFrame(sonuclar)


# ──────────────────────────────────────────────
# İSTATİSTİK TABLOSU  (Talimatname Bölüm 5)
# ──────────────────────────────────────────────

def istatistik_tablosu(df: pd.DataFrame):
    """
    Talimatname Bölüm 5 tablosunu ekrana ve CSV'e yazar:
        Sınıf | Örnek Sayısı | Ort. F0 (Hz) | Std Sapma | Başarı (%)
    """
    siniflar = ['Erkek', 'Kadın', 'Çocuk']

    print('\n' + '═' * 65)
    print('İSTATİSTİKSEL TABLO  (Talimatname Bölüm 5)')
    print('═' * 65)
    print(f'{"Sınıf":<10} {"Örnek":<8} {"Ort. F0 (Hz)":<16} '
          f'{"Std Sapma":<14} {"Başarı (%)"}')
    print('─' * 65)

    satirlar = []
    for sinif in siniflar:
        alt = df[df['Gercek'] == sinif]
        if len(alt) == 0:
            continue
        n      = len(alt)
        ort_f0 = alt['Ortalama_F0_Hz'].mean()
        std_f0 = alt['Ortalama_F0_Hz'].std()
        dogru  = alt['Dogru'].sum() if alt['Dogru'].notna().any() else 0
        basari = 100 * dogru / n

        print(f'{sinif:<10} {n:<8} {ort_f0:<16.2f} {std_f0:<14.2f} %{basari:.1f}')
        satirlar.append({
            'Sınıf': sinif, 'Örnek_Sayısı': n,
            'Ortalama_F0_Hz': round(ort_f0, 2),
            'Std_Sapma': round(std_f0, 2),
            'Basari_Yuzde': round(basari, 1)
        })

    etiketli = df.dropna(subset=['Dogru'])
    if not etiketli.empty:
        genel = 100 * etiketli['Dogru'].sum() / len(etiketli)
        print('─' * 65)
        print(f'{"GENEL":<10} {len(etiketli):<8} {"—":<16} {"—":<14} %{genel:.1f}')
    print('═' * 65)

    # CSV kaydet
    os.makedirs(CIKTI_KLASOR, exist_ok=True)
    tablo_yolu = os.path.join(CIKTI_KLASOR, 'istatistik_tablosu.csv')
    pd.DataFrame(satirlar).to_csv(tablo_yolu, index=False, encoding='utf-8-sig')
    print(f'[KAYIT] {tablo_yolu}')


# ──────────────────────────────────────────────
# HATA ANALİZİ  (Talimatname Bölüm 5)
# ──────────────────────────────────────────────

def hata_analizi(df: pd.DataFrame):
    """Yanlış sınıflandırılan dosyaları listeler ve olası nedenleri yorumlar."""
    hatalar = df[(df['Dogru'] == False)].copy()

    if hatalar.empty:
        print('\n✅ Hata yok — tüm dosyalar doğru sınıflandırıldı.')
        return

    print(f'\n{"─"*65}')
    print(f'HATA ANALİZİ  ({len(hatalar)} yanlış tahmin)')
    print(f'{"─"*65}')
    print(f'{"Dosya":<35} {"Gerçek":<8} {"Tahmin":<8} {"F0":>8}  Olası Neden')
    print('─' * 65)

    for _, r in hatalar.iterrows():
        # Olası neden tahmini
        neden = _neden_tahmin(r)
        print(f'{r["Dosya_Adi"]:<35} {str(r["Gercek"]):<8} {r["Tahmin"]:<8} '
              f'{r["Ortalama_F0_Hz"]:>8.1f}  {neden}')


def _neden_tahmin(satir) -> str:
    """Hata için kısa teknik yorum üretir."""
    f0    = satir['Ortalama_F0_Hz']
    duygu = str(satir.get('Duygu', '')).lower()

    if 'angry' in duygu or 'sinirli' in duygu:
        return 'Öfke duygusu F0\'ı yükseltmiş olabilir'
    if f0 > 200 and satir['Gercek'] == 'Erkek':
        return 'Yüksek F0 — genç erkek veya duygusal konuşma'
    if f0 < 200 and satir['Gercek'] == 'Çocuk':
        return 'Düşük F0 — büyük çocuk / ergenlik sesi'
    if satir['Gercek'] == 'Kadın' and satir['Tahmin'] == 'Çocuk':
        return 'F0 geçiş bölgesinde, yüksek ZCR etkisi'
    return 'F0 sınır bölgesinde — ayrım güç'


# ──────────────────────────────────────────────
# TEK DOSYA TAHMİNİ
# ──────────────────────────────────────────────

def tek_dosya_tahmin(dosya_yolu: str):
    """Tek bir wav dosyası için konsol çıktısı ile tahmin yapar."""
    print(f'\n[TAHMİN] {os.path.basename(dosya_yolu)}')
    ozellikler = extract_features(dosya_yolu)

    if ozellikler is None:
        print('  Öznitelik çıkarılamadı. Ses dosyasını kontrol edin.')
        return

    tahmin = siniflandir(ozellikler)
    emoji  = {'Erkek': '👨', 'Kadın': '👩', 'Çocuk': '👶'}.get(tahmin, '?')

    print(f'  ┌─────────────────────────────')
    print(f'  │  Tahmin   :  {emoji}  {tahmin}')
    print(f'  │  Ort. F0  :  {ozellikler["mean_f0"]:.2f} Hz')
    print(f'  │  Std F0   :  {ozellikler["std_f0"]:.2f} Hz')
    print(f'  │  ZCR      :  {ozellikler["mean_zcr"]:.4f}')
    print(f'  │  STE      :  {ozellikler["mean_ste"]:.6f}')
    print(f'  └─────────────────────────────')

    # Grafikleri de üret
    plot_acf_vs_fft(dosya_yolu)
    plot_ste_zcr(dosya_yolu)


# ──────────────────────────────────────────────
# ANA ÇALIŞMA AKIŞI
# ──────────────────────────────────────────────

def main():
    os.makedirs(CIKTI_KLASOR, exist_ok=True)

    # ── Tek dosya modu ──────────────────────────
    if len(sys.argv) > 1 and sys.argv[1].endswith('.wav'):
        tek_dosya_tahmin(sys.argv[1])
        return

    # ── Tam veri seti analizi ───────────────────
    print('╔══════════════════════════════════════════════════════════╗')
    print('║   SES SİNYALİ ANALİZİ VE CİNSİYET SINIFLANDIRMA        ║')
    print('║   2025-2026 Bahar Dönemi — Dönemiçi Proje               ║')
    print('╚══════════════════════════════════════════════════════════╝\n')

    # 1. Veriyi oku (Excel + os.path.exists kontrolü)
    master = veri_oku(DATASET_KLASOR)

    # 2. Analiz döngüsü
    sonuclar = analiz_pipeline(master)

    if sonuclar.empty:
        print('[HATA] Sonuç üretilemedi.')
        return

    # 3. İstatistik tablosu
    istatistik_tablosu(sonuclar)

    # 4. Hata analizi
    hata_analizi(sonuclar)

    # 5. Grafikler
    print('\n[GRAFİK] Görseller üretiliyor...')

    # Otokorelasyon vs FFT grafiği için ilk geçerli dosyayı kullan
    ilk_dosya = sonuclar['Dosya_Yolu'].iloc[0]
    f0_acf, f0_fft = plot_acf_vs_fft(ilk_dosya)
    if f0_acf and f0_fft:
        fark = abs(f0_acf - f0_fft)
        print(f'  ACF F0={f0_acf:.1f} Hz  |  FFT F0={f0_fft:.1f} Hz  |  Fark={fark:.1f} Hz')

    plot_f0_dagilimi(sonuclar)
    plot_confusion_matrix(sonuclar)

    # 6. Sonuçları CSV olarak kaydet
    csv_yolu = os.path.join(CIKTI_KLASOR, 'tahmin_sonuclari.csv')
    sonuclar.to_csv(csv_yolu, index=False, encoding='utf-8-sig')
    print(f'[KAYIT] {csv_yolu}')

    print(f'\n✅ Tüm çıktılar  →  {CIKTI_KLASOR}/  klasörüne kaydedildi.')


if __name__ == '__main__':
    main()
