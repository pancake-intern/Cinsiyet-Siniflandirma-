# aCinsiyet Sınıflandırma Projesi 🎙️

Ses sinyallerinden **Erkek / Kadın / Çocuk** cinsiyetini otomatik olarak tahmin eden kural tabanlı bir sınıflandırıcı.

---

## 📌 Proje Özeti

Bu proje, ses kayıtlarından çıkarılan akustik öznitelikler (F0, ZCR, Enerji) kullanılarak konuşmacının cinsiyetini sınıflandırmayı amaçlamaktadır. Temel fundamental frekans (F0) eşik değerleri veri setinden istatistiksel olarak hesaplanmış ve kural tabanlı bir sınıflandırıcı tasarlanmıştır.

---

## 🗂️ Dosya Yapısı

```
Cinsiyet-Siniflandirma-/
├── MidtermProject.ipynb   # Ana Jupyter Notebook
├── Midterm_Dataset_2026/  # Ses veri seti (Grup_01, Grup_02, ...)
├── MetaData.xlsx          # Birleştirilmiş metadata dosyası
└── README.md
```

---

## ⚙️ Kullanılan Teknolojiler

| Kütüphane | Amaç |
|-----------|------|
| `librosa` | Ses yükleme ve sinyal işleme |
| `numpy` / `scipy` | Sayısal hesaplamalar, otokorelasyon |
| `pandas` | Veri yönetimi ve metadata işleme |
| `matplotlib` / `seaborn` | Görselleştirme |
| `scikit-learn` | Confusion matrix ve doğruluk metrikleri |

---

## 🔬 Yöntem

### 1. Öznitelik Çıkarımı
Her ses dosyasından aşağıdaki öznitelikler çıkarılmaktadır:
- **F0 (Fundamental Frequency / Temel Frekans):** Otokorelasyon yöntemiyle hesaplanır.
- **ZCR (Zero Crossing Rate / Sıfır Geçiş Oranı):** Ses sinyalinin işaret değiştirme hızı.
- **Kısa Süreli Enerji (STE):** Her penceredeki normalize edilmiş enerji değeri.

### 2. F0 Hesaplama Yöntemleri
- **Otokorelasyon (Ana Yöntem):** `R(τ) = Σ x[n] · x[n − τ]` formülüyle periyot tahmini yapılır.
- **FFT (Karşılaştırma Amacıyla):** Büyüklük spektrumundan ilk baskın harmonik bulunur.

### 3. Kural Tabanlı Sınıflandırıcı

| Sınıf | F0 Aralığı |
|-------|------------|
| Erkek | F0 < 185 Hz |
| Kadın | 185 Hz ≤ F0 < 300 Hz |
| Çocuk | F0 ≥ 300 Hz |

> Eşik değerleri, veri setinin F0 istatistiklerinden türetilmiştir:
> - Erkek Ort ≈ 186 Hz
> - Kadın Ort ≈ 270 Hz
> - Çocuk Ort ≈ 335 Hz

---

## 📊 Değerlendirme

- **Genel Doğruluk (Accuracy):** Sınıflandırıcı genel başarısı
- **Sınıf Bazlı Rapor:** Precision, Recall, F1-Score
- **Confusion Matrix:** Gerçek ve tahmin edilen sınıfların karşılaştırması

---

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler

```bash
pip install librosa numpy pandas matplotlib seaborn scipy scikit-learn openpyxl
```

### Çalıştırma

1. Veri setini `Midterm_Dataset_2026/` klasörüne yerleştirin.
2. `MetaData.xlsx` dosyasının proje dizininde olduğundan emin olun.
3. Jupyter Notebook'u açın ve hücreleri sırayla çalıştırın:

```bash
jupyter notebook MidtermProject.ipynb
```

---

## 📁 Veri Seti Formatı

`MetaData.xlsx` dosyasında aşağıdaki sütunlar bulunmalıdır:

| Sütun | Açıklama |
|-------|----------|
| `File name` | Ses dosyasının adı |
| `Subject_ID` | Katılımcı kimliği |
| `Gender` | Cinsiyet (E / K / C) |
| `Age` | Yaş |
| `Feeling` | Duygu durumu |
| `Sentence_No` | Cümle numarası |
| `Recording_Device` | Kayıt cihazı |
| `ENVIRONMENT` | Kayıt ortamı |
| `noise level` | Gürültü seviyesi |
