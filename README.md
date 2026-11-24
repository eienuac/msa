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

`--control` bayrağı tuşları düz SendInput ile gönderir. Hiçbir gizleme veya anti-cheat atlatma içermez. Yalnızca otomasyona izin verilen ortamlarda kullanın. F12 ajanı durdurur ve tüm tuşları bırakır. Tuşlar yalnızca `--window-title` ile verilen pencere öndeyken gönderilir.

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

## Bilinen eksikler (sıradaki adımlar)

- Boss'un **yönü** okunmuyor; mesafe biliniyor, yön "ileride" varsayılıyor. Oku okuyan bir glif dedektörü ya da boss dedektörü gerekiyor.
- Çağırma saldırısından sonra hedefin yan yaratığa geçtiği OCR ile anlaşılıyor, ama ajanın boss'u yeniden seçmesi için "boss'un yanına git + R" dışında bir yol modellenmedi.
- Gölge modda tuşa basılmadığı için makro durum makinesi basışların "kaybolduğunu" görüp kendini kilitliyor. Bu beklenen bir durum; canlı kontrolde oluşmaz.
- Ekranda birbirine değen birden fazla daire tek bölge olarak birleşiyor; daireleri tek tek ayırma gerekiyor.
- Kamera modelinin ölçeği varsayıma dayanıyor. Bir kalibrasyon kaydıyla netleşecek.
- Görüntü işleme tek çekirdekte ~33 ms/kare sürüyor. Hedef ≤15 ms.
