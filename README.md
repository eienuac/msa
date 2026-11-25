# Boss AI Agent

Pekiştirmeli öğrenme (MaskablePPO) ile eğitilen, oyun ekranını görüntü işleme ile okuyan bir boss savaşı ajanı.

Ajan piksel görmez. Görüntü işleme ekranı bir `GameState` nesnesine çevirir. Simülasyon da aynı `GameState`'i üretir. Ajan simülasyonda eğitilir, gerçek oyunda aynı arayüzle çalışır.

```
ekran ─► perception (HUD, tehlike tespiti, takip, zemine izdüşüm) ─┐
                                                                    ├─► GameState ─► encoder ─► PPO ─► niyet
simülasyon (dünya + algı gürültüsü + gecikme) ──────────────────────┘                                  │
                                         tuşlar ◄─ ActionController ◄─ MacroStateManager (R) ◄─────────┘
```

## Klasörler

| Yol | İçerik |
|---|---|
| `bossai/core/` | `GameState` sözleşmesi, tehlike şekilleri (daire, yelpaze, çizgi, poligon), tehlike ızgarası, zaman-farkındalıklı kaçış ışınları ve planlayıcı |
| `bossai/control/` | R makrosu durum makinesi, tuş kontrolcüsü, giriş arka uçları (kayıt / boş / Windows SendInput) |
| `bossai/agent/` | Aksiyon uzayı ve maskeleri, gözlem kodlayıcı (ızgara + vektör), CNN+MLP politika ağı |
| `bossai/sim/` | 2D boss simülasyonu, seviyeler (curriculum), algı gürültüsü modeli, Gymnasium ortamı, kuşbakışı çizim |
| `bossai/rewards/` | Ödül fonksiyonu (bileşenler ayrı ayrı loglanır) |
| `bossai/training/` | Eğitim betiği, curriculum ve loglama callback'leri, model klasörü/şema kontrolü |
| `bossai/evaluation/` | Sabit seed'li senaryo değerlendirmesi, betik taban çizgileri, başarısızlık kaydı, video |
| `bossai/perception/` | HUD okuyucu, tehlike tespiti, takip ve patlama zamanı tahmini, kamera modeli, görüntü hattı |
| `bossai/capture/`, `bossai/runtime/` | Ekran yakalama (dxcam / mss / video dosyası), canlı/gölge mod döngüsü |
| `tools/analyze_video.py` | Kayıtlı videoya tespitleri çizer, CSV zaman çizelgesi çıkarır |
| `configs/` | Boss'a özel simülasyon ayarları |

## Kurulum

Python 3.11+ gerekir.

```bash
pip install -e .[dev]
```

Windows'ta canlı ekran yakalama için:

```bash
pip install -e .[windows]
```

Boss mesafesini ve hedef adını okumak için (RapidOCR, modelleri paketle gelir):

```bash
pip install -e .[ocr]
```

## Komutlar

Testler:

```bash
pytest
```

Eğitim (curriculum 0 → 4 otomatik ilerler):

```bash
python -m bossai.training.train --timesteps 3000000 --n-envs 8 --run-name kor_bekci_v1
```

Eğitim grafikleri:

```bash
tensorboard --logdir models/kor_bekci_v1/tb
```

Değerlendirme (taban çizgileriyle karşılaştırma, en kötü bölümün videosu):

```bash
python -m bossai.evaluation.evaluate --model models/kor_bekci_v1 --levels 0 2 4 --episodes 20 --baselines --video worst
```

Simülasyonu yukarıdan bakış videosu olarak izleme:

```bash
python tools/render_sim.py --model models/kor_bekci_v2 --level 3 --seconds 90 --out sim.mp4
```

Kayıtlı bir videoda görüntü işlemeyi test etme:

```bash
python tools/analyze_video.py kayit.mp4 --out analysis
```

Gölge mod: ajan tuşa basmaz, ne yapacağını ekrana çizer.

```bash
python -m bossai.runtime.live --model models/kor_bekci_v1 --source kayit.mp4 --out golge.mp4
```

```bash
python -m bossai.runtime.live --model models/kor_bekci_v1 --source dxcam --show
```

`--control` bayrağı tuşları düz SendInput ile gönderilmemeli ve arkasında iz bırakmamalı

## Örümcek Ana modeli: `pretrained/orumcek_v1`

Kor Bekçi v4 modelinden başlayıp örümcek simülasyonunda eğitildi (3 milyon adım, curriculum 0→4, bölümlerin %25'i daha kolay seviyelerden). Takip eden halkaya karşı E kalkanı kural tabanlı bir reflekstir; politika ona karışmaz. 12 sabit senaryo (simülasyon):

| Seviye | Politika | Kazanma | Ölüm | Öldürme süresi | İsabet/dk | Erken kaçış | R/dk |
|---|---|---|---|---|---|---|---|
| 4 (en zor) | **PPO** | **%75** | %25 | 156 sn | 1,89 | %79 | 9,6 |
| 4 | kural tabanlı uzman | %42 | %58 | 185 sn | 2,64 | %70 | 6,7 |
| 3 | **PPO** | **%100** | %0 | 153 sn | 0,91 | %84 | 6,0 |
| 3 | kural tabanlı uzman | %83 | %17 | 173 sn | 1,65 | %75 | 3,2 |
| 2 | **PPO** | **%100** | %0 | 141 sn | 0,92 | %89 | 7,1 |
| 2 | kural tabanlı uzman | %100 | %0 | 155 sn | 1,19 | %85 | 3,4 |

Varsayımlar (kullanıcı henüz doğrulamadı): kalkanın bekleme süresi 3-5 sn, zehirde iki darbe arası 1,0-1,5 sn, koza sersemletmesi 1,0-1,5 sn. Bunlar değişirse kısa bir ek eğitim gerekir.

```bash
python -m bossai.runtime.live --model pretrained/orumcek_v1 --boss orumcek_ana --source kayit.mp4 --out golge.mp4
```

## Eğitilmiş model: `pretrained/kor_bekci_v4` (güncel)

v3'ün üzerine 1 milyon adım, düzeltilmiş makro davranışıyla: yürümek makroyu her zaman değil, bazen kapatır (kayıt 33/44). Değerlendirme yeni simülasyonda, 15 görülmemiş senaryoda, seviye 4:

| Politika | Kazanma | Ölüm | Öldürme süresi | İsabet/dk | Erken kaçış | Dash/dk | R/dk |
|---|---|---|---|---|---|---|---|
| **PPO v4** | %93 | %0 | 155 sn | 0,40 | %87 | 12,5 | 5,5 |
| PPO v3 | %93 | %0 | 160 sn | 0,60 | %86 | 12,0 | 5,2 |
| kural tabanlı uzman | %100 | %0 | 159 sn | 0,40 | %85 | 11,0 | 7,0 |

R basışı eski simülasyondaki ~15/dk'dan ~5/dk'ya indi. Düşüşün çoğu simülasyonun düzeltilmesinden ve durum makinesinin R'ye sadece gösterge "kapalı" derken basmasından geliyor (v3 de yeni simülasyonda ~5/dk basıyor). Bu ayarlarda uzman politika kazanma oranında PPO'yu yakaladı; PPO 15 savaştan 1'ini süre dolduğu için kaybetti.

## Önceki model: `pretrained/kor_bekci_v3`

3,2 milyon adım (MaskablePPO, curriculum 0→4). Değerlendirme, eğitimde görülmemiş 15 sabit senaryo üzerinde yapıldı (simülasyon):

| Seviye | Politika | Kazanma | Ölüm | Öldürme süresi | İsabet/dk | Erken kaçış | Dash/dk | R/dk |
|---|---|---|---|---|---|---|---|---|
| 4 (en zor) | **PPO ajanı** | **%100** | %0 | 155 sn | 0,21 | %88 | 12,4 | 14,9 |
| 4 | kural tabanlı uzman | %73 | %0 | 187 sn | 0,44 | %89 | 10,7 | 19,5 |
| 4 | hiç kaçmayan | %0 | %100 | – | 12,3 | – | 0 | 3,9 |
| 3 | **PPO ajanı** | %100 | %0 | 149 sn | 0,48 | %88 | 9,7 | 12,0 |
| 3 | kural tabanlı uzman | %100 | %0 | 162 sn | 0,37 | %88 | 8,7 | 18,2 |

Bunlar simülasyon sonuçları; gerçek oyunda henüz test edilmedi.

v4 için `--model pretrained/kor_bekci_v4` verin.

## Videolardan ölçülenler (Kor Bekçi)

- **Uyarı süresi:** daire ve yelpaze için 1,1–1,5 sn. Boss'un saldırıları arasında ~7 sn var.
- **Uyarının yapısı:** dış sınır bir anda tam boyutta beliriyor, içteki koyu dolgu merkezden (yelpazede tepeden) kenara büyüyor, kenara değince patlıyor.
- **Hasar:** isabet başına canın %12–25'i.
- **Dash:** 3 şarj. Dashler arası kilit 1,0 sn ("1,0" geri sayımı). Bir şarj ~3,3 sn'de doluyor. Üçüncü dashten sonra ~8 sn bekleme. Dash atarken hasar alınabiliyor (dokunulmazlık yok).
- **R makrosu:** açıkken ikonun çerçevesi sarı (sarı piksel oranı açıkken 0,09, kapalıyken 0,000). İkon da daha parlak.
- **Boss barı:** ad, mesafe ve yön ("▲ 5 m") ile HP sayıları görünüyor. Bar okuması sayılarla ±%1,5 uyumlu.
- **Ateş patlaması (çağırma):** boss kollarını kaldırır, ~1 sn sonra etrafında 4–5 ateş patlaması olur. Hasar vermez; etrafa yaratık çıkarır ve oyuncuyu biraz geri savurur. Çıkan yaratıklar hedefi boss'tan alabilir.
- **Makro:** yürümek makroyu kapatır. Bu yüzden ajan yürürken R'ye hiç basmaz; durduktan sonra tek basışla makroyu yeniden açar. Boss'un yanında R'ye basmak boss'u yeniden hedef alır.

## Kaçış tercihi

Ajan son saniyeyi beklememeli. Ödülde bunun için üç terim var:

- Uyarı alanının içinde geçen her 0,1 sn için ceza.
- Patlamaya 0,5 sn'den az kala çıkılan kaçışlar "geç kaçış" sayılır ve cezalandırılır.
- Sadece en az 0,5 sn payla yapılan kaçışlar ödüllendirilir.

"Güvenli bölgeye git" aksiyonu da en az 0,5 sn pay bırakan çıkışları seçer.

## Görüntüden okunanlar

| Bilgi | Yöntem | Doğrulama (kayıtlar) |
|---|---|---|
| Makro açık/kapalı | R çerçevesindeki sarı oranı + ikon parlaklığı | açık 1,00 / kapalı 0,00 |
| Oyuncu / boss HP | Bar doluluk oranı | boss barı sayılarla ±%1,5 |
| Dash durumu | Q ikonunun rengi + küçük şarj rakamı (şablon eşleme) | 4 videoda tüm şarj/kilit geçişleri |
| Boss mesafesi, hedef adı | RapidOCR + son 3 okumanın ortancası | tek karede 30/33 doğru |
| Tehlike alanları | Renk tonu + dışbükey bölge + takip + dolum hızı | 4 uyarıda patlama ±0,3 sn |
| Oyuncunun ekrandaki yeri | Kafa üstündeki kırmızı can + mavi mana barı; zemin izdüşümü buna göre kayar | kamera yürürken/dash'te 30-80 px geride kalıyor; bar 6 kaydın hepsinde dinlenme noktasında (960, 512) |
| Kamera eğimi | Yerdeki daireler ancak doğru eğimle zemine geri çizilince daire olur; savaş boyunca biriktirilir, eski kanıt 30 sn'de yarıya iner | Kor Bekçi savaş 1: 19,4°, savaş 2: 23,4° (tek tek daireler ±1°) |

## Boss profilleri

Her boss'un görüntü ayarları `configs/bosses/<id>.yaml` dosyasında: uyarı tarzı, renk eşikleri, OCR'da hedef adının parçası ve tipik uyarı süresi. Araçlara `--boss <id>` verilir.

| Profil | Uyarı tarzı | Durum |
|---|---|---|
| `kor_bekci` | `filled`: kırmızı alan, içten dolan dolgu | ölçüldü, simülasyon ve model hazır |
| `orumcek_ana` | `rim`: soluk yeşil-sarı yarı saydam disk, sadece kenar halkası belirgin; zemin turuncu kum | tespit var; mekanikler kullanıcıdan (zehir iki aşamalı, takip eden halkaya E kalkanı), simülasyon profili ve `pretrained/orumcek_v1` hazır |

```bash
python tools/analyze_video.py kayit.mp4 --boss orumcek_ana --out analysis
```

## Bilinen eksikler (sıradaki adımlar)

- Boss'un **yönü** okunmuyor; mesafe biliniyor, yön "ileride" varsayılıyor. Oku okuyan bir glif dedektörü ya da boss dedektörü gerekiyor.
- Çağırma saldırısından sonra hedefin yan yaratığa geçtiği OCR ile anlaşılıyor, ama ajanın boss'u yeniden seçmesi için "boss'un yanına git + R" dışında bir yol modellenmedi.
- Gölge modda tuşa basılmadığı için makro durum makinesi basışların "kaybolduğunu" görüp kendini kilitliyor. Bu beklenen bir durum; canlı kontrolde oluşmaz.
- Kamera eğimi yalnızca tam görünen dairelerden ölçülüyor; ilk dairelere kadar varsayılan eğim (19°) kullanılıyor. Yelpaze ve çizgi saldırıları katkı vermiyor. Örümceğin kenar halkaları çoğunlukla yarım yay olarak göründüğü için eğimi net belirlemiyor; filtre bunları reddediyor, örümcekte varsayılan eğim kalıyor.
- Kameranın sağ tıkla **yatay** döndürülmesi zemin ölçümünü bozmaz (zemin koordinatları kameraya göre), ama boss'un yönü kamera yönüne göre okunmalı.
- Görüntü işleme tek çekirdekte Kor Bekçi'de ~11-15 ms/kare, örümcekte ~28 ms/kare (kenar halkası maskesi OpenCV'nin doymalı uint8 işlemleriyle; numpy ile 106 ms idi).
