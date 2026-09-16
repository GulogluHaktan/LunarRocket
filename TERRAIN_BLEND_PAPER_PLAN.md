# Ay İnişi Simülasyonunda Arazi Detay-Geçiş Yöntemleri — Tam Makale Planı

Durum: LunarRocket / Isaac Lab tabanlı ay iniş simülatörü henüz tamamlanmadı (RL eğitimi,
tam sistem sonuçları yok). Bu yüzden makale **tüm sistemi değil**, sistemin şu an en olgun ve
kendi başına ölçülebilir parçasını — arazi (terrain) üretim/detay-geçiş alt sistemini — konu
alıyor. Hedef: 8-12 sayfa, konferans/TR dizin, hızlı üretilebilir, aşırı komplike olmayan.

Bu dosya bir sonraki Claude Code oturumunda doğrudan uygulanabilecek kadar detaylı, uçtan uca
bir plandır: fikir → literatür konumlandırması → yöntemler (formüllerle) → deney matrisi →
metrikler → makale iskeleti → uygulama checklist'i → zaman planı.

---

## 1. Makale Fikri (tek cümle)

Isaac Lab tabanlı itkili ay iniş simülasyonunda, iniş bölgesi çevresinde çözünürlük/temas
gerçekçiliği ile VRAM/hesaplama maliyeti arasındaki dengeyi yöneten üç arazi detay-geçiş
yöntemini (mevcut smoothstep-tabanlı iki-mesh sistemi, klasik Gaussian karışım, ve bölgesel
hibrit) aynı mimari üzerinde kontrollü biçimde karşılaştırıyor ve nicel sonuçlar sunuyoruz.

## 2. Neden bu konu, neden şu an

- Sistemin geri kalanı (RL eğitimi, gözlem/ödül tasarımı) henüz olgunlaşmadı; onları makaleye
  koymak "bitmemiş sistem" izlenimi verir.
- Terrain alt sistemi kendi başına tamamlanmış, çalışan bir bileşen: `app/terrain_generator.py`,
  `app/benchmark_terrain_quality.py` zaten üretim/ölçüm yapabiliyor.
- Konu dar ve somut olduğu için 1 hafta içinde deney + yazım tamamlanabilir.
- Deadline (22 Ağustos 13:30) için sadece fikir isteniyor — bu doküman o fikrin arkasındaki
  tam planı taşıyor, gerekirse doğrudan gruba da özetlenebilir.

## 3. Şu an kodda ne var (mevcut durum tespiti)

`app/terrain_generator.py` içinde üç ilgili yapı var:

### 3.1 Yöntem A — fiilen çalışan yol
`generate()` → `generate_local_detail_mesh()` (satır ~517-561). Akış:
1. Global (kaba) terrain normal şekilde üretilir → tek USD mesh (`TerrainBase`).
2. İniş bölgesi için **ayrı, bağımsız, yüksek çözünürlüklü bir mesh** üretilir:
   - Global mesh'ten bilinear örnekleme ile temel yükseklik alınır (`_sample_base_heights_to_local`).
   - Üstüne krater, çok-ölçekli gürültü, regolith tane deseni, mikro-özellik, footprint detayı eklenir.
   - Kenarda `_local_detail_mask()` (satır ~563-574) ile **smoothstep** maskesi uygulanır:
     `mask(t) = t² (3 - 2t)`, `t = clip(distance_to_edge / fade_width_m, 0, 1)`.
3. İki mesh, sahnede aynı konumda üst üste, **ayrı collision mesh** olarak yerleştirilir
   (`create_or_update_usd_meshes`, satır ~242-264).

**Özellikler**: kompakt destek (fade_width dışında tam 0), C¹ süreklilik (uçlarda türev 0),
gerçek yüksek çözünürlük korunur (ayrı mesh olduğu için).

### 3.2 Yöntem B — dokümante edilmiş ama ÇAĞRILMAYAN (dead code)
`generate_local_detail_patch()` (satır ~715-764) + `_blend_local_patch_multiscale()`
(satır ~924-991). Formül:
```
blend_factor(d) = exp(-(d / blend_idx)^blend_scale)     # blend_scale=2.0 (config)
result[i,j] = global_h[i,j] + blend_factor * local_value(i,j)   # bilinear örneklenmiş local değer
```
`TWO_TIER_LANDING.md`'deki tablo (4m: %90, 8m: %36, 12m: %5, 16m: %0) bu formülden.

**Kritik sorun**: `result = global_h.copy()` — yani çıktı **global grid çözünürlüğünde**
kalıyor (örn. 128×128). Local'in 256×256 detayı buraya "blend" edilse bile üretilen mesh'in
vertex/üçgen sayısı artmıyor, sadece var olan kaba vertex'lerin yükseklik *değeri* değişiyor.
Yani bu yöntem, iddia ettiği "130k üçgen iniş bölgesinde" hedefini **matematiksel olarak
sağlayamaz**. Bu, terk edilmesinin muhtemel gerçek sebebi.

**Makaledeki rolü**: doğrudan "rakip yöntem" olarak sunulmamalı (adil değil, çözünürlük
kaybı yüzünden zaten dezavantajlı) — ama "neden A'yı seçtik" gerekçesi olarak, ve "naif tek-grid
Gaussian karışımın neden yetersiz kaldığı" başlıklı kısa bir alt-bölüm olarak değerli.
Bkz. Yöntem A2 (§4.2) — adil karşılaştırma için düzeltilmiş versiyonu.

### 3.3 Konfigürasyon (mevcut, `configs/terrain_config.yaml`)
```yaml
local_detail_patch:
  enabled: true
  patch_size_m: [30.0, 30.0]
  resolution: 256
  blend_radius_m: 8.0      # Yöntem B'nin "r" parametresi (A'da fade_width_m karşılığı)
  blend_scale: 2.0         # Yöntem B'nin "p" parametresi
  noise_scale: 0.015
  noise_smoothing_passes: 1
  edge_fade_width_m: 4.0   # Yöntem A'nın gerçekte kullandığı parametre (_local_detail_mask)
```

## 4. Karşılaştırılacak Yöntemler (deney tasarımı)

Aşama 1'in temel ilkesi: **mimariyi sabit tut, sadece kenar-geçiş fonksiyonunu değiştir.**
Böylece çözünürlük, üçgen sayısı, VRAM gibi değişkenler kontrol altında kalır ve tek fark
gerçekten karşılaştırılmak istenen şey (fonksiyonun şekli) olur.

### 4.1 Ortak mimari (tüm yöntemlerde sabit)
- İki-mesh yapı korunur (global + local, Yöntem A'nın mimarisi).
- Local mesh çözünürlüğü, patch boyutu, gürültü/krater parametreleri sabit.
- Tek değişken: `_local_detail_mask()` içindeki fonksiyon.

### 4.2 Yöntem A2 — Smoothstep (mevcut, baseline)
```
mask(d) = t²(3-2t),  t = clip((fade_width_m - d) / fade_width_m, 0, 1)
```
Zaten kodda var, değişiklik gerekmiyor — sadece `edge_fade_function: smoothstep` etiketiyle
"kontrol grubu" olarak işaretlenecek.

### 4.3 Yöntem B2 — Gaussian (düzeltilmiş, adil versiyon)
Yöntem B'nin formülünü al ama **aynı iki-mesh mimarisine** uygula (global grid'e değil, local
mesh'in kendi edge-mask'ine):
```
mask(d) = exp(-(d / blend_radius_m)^blend_scale),  sonra normalize: mask_norm = mask / mask(0)
```
`blend_radius_m` ve `blend_scale` zaten config'te mevcut. Not: Gaussian sonsuz destekli
olduğundan, `fade_width_m`'nin dışında kalan bölgede küçük bir kalıntı (residual) olabilir —
bunu ya bir eşikte (`mask < 0.001`) sert kes, ya da raporda bu "truncation" maliyetini ayrı bir
metrik olarak göster (bilimsel dürüstlük açısından önemli — Yöntem B'nin eleştirisini A2 için
tekrarlamamak adına).

### 4.4 Yöntem C — Hibrit (asıl katkı, Aşama 2)
İniş takımının değeceği dar iç bölgede Gaussian, onun dışında smoothstep:
```
mask(d) = gaussian(d)     if d < r_inner   (örn. r_inner = 1.5m, iniş ayağı izi yarıçapı)
mask(d) = smoothstep(d)   if r_inner <= d <= fade_width_m
```
Süreklilik için: `r_inner` noktasında iki fonksiyonun değerini eşitleyecek şekilde Gaussian'ın
genlik/ölçek parametresi kalibre edilir (basit bir sabit çarpan yeterli, C⁰ süreklilik hedefi;
C¹ istenirse `blend_scale` ince ayarı ile yaklaşık sağlanabilir — bu isteğe bağlı bir
"nice-to-have", zorunlu değil).

**Hipotez**: C, temas-kritik dar bölgede (iniş ayağı altı) Gaussian'ın pürüzsüzlüğünden
faydalanırken, geniş alanda A2'nin öngörülebilir/sınırlı davranışını korur — yani ikisinin de
tek başına sahip olduğu dezavantajı (A2: merkeze yakın nispeten sert geçiş; B2: sonsuz destek/
truncation maliyeti) azaltır.

## 5. Metrikler ve Ölçüm Planı

| # | Metrik | Nasıl ölçülür | Hangi yöntem(ler) için anlamlı |
|---|--------|----------------|-------------------------------|
| 1 | Üretim süresi (ms) | `benchmark_terrain_quality.py` zaten ölçüyor, `time.perf_counter()` | A2, B2, C |
| 2 | VRAM (MB) | Mevcut script (`torch.cuda.memory_allocated` veya `nvidia-smi`) | A2, B2, C |
| 3 | Üçgen/vertex sayısı | Mevcut `TerrainMeshData` alanları | A2, B2, C (kontrol amaçlı — üçünde de eşit olmalı, sanity check) |
| 4 | Kenar sürekliliği | Sınırdaki yükseklik alanının ikinci türevinin (Laplacian) normu, `fade_width_m` civarında örneklenmiş | A2 vs B2 vs C |
| 5 | Truncation kalıntısı | Gaussian mask'in `fade_width_m` noktasındaki değeri (idealde ~0) | B2, C (Gaussian kısmı) |
| 6 | Temas doğruluğu (opsiyonel, Isaac Sim gerektirir) | Roket iniş takımının zemine batma derinliği (penetration depth), birkaç sabit seed üzerinden ortalama ± std | A2, B2, C |
| 7 | Tutarlılık | Aynı seed ile üretilen terrain'in min/max/ortalama eğim istatistikleri (`mean_slope_deg`, `max_slope_deg`, zaten `TerrainMeshData`'da var) | A2, B2, C |

### Deney matrisi
- **Preset'ler**: mevcut `low_debug`, `medium_train`, `high_train`, `high_detail_landing`
  (4 preset × 3 yöntem = 12 koşu).
- **Seed'ler**: her hücre için en az 5 farklı seed (varyans/std raporlamak için) →
  toplam 60 koşu. Isaac Sim gerektirmeyen metrikler (1-5, 7) için bu CPU'da bile hızlı çalışır.
- Metrik 6 (temas doğruluğu) opsiyonel — GPU/Isaac Sim erişimi kısıtlıysa sadece
  `high_detail_landing` preset'inde, 3 seed ile sınırlı tutulabilir (zaman kısıtı için).

## 6. Literatür Konumlandırması (önceki araştırmadan, kaynaklarla)

- **Genel çok-çözünürlüklü terrain**: Losasso & Hoppe, *Geometry Clipmaps: Terrain Rendering
  Using Nested Regular Grids*, SIGGRAPH/ACM TOG 2004. Standart, eski bir teknik — "biz icat
  ettik" denemez.
- **Gezegen inişi için çok-çözünürlüklü elevation mapping**: *Multi-Resolution Elevation
  Mapping and Safe Landing Site Detection with Applications to Planetary Rotorcraft*
  (arXiv 2111.06271).
- **Ay-spesifik, Isaac Sim tabanlı en yakın çalışma — OmniLRS**: *OmniLRS: A Photorealistic
  Simulator for Lunar Robotics* (arXiv 2309.08997, Luxembourg/Tohoku). Gerçek DEM (5m/pixel) +
  prosedürel ince detay (2.5cm/pixel), geometry clipmap içinde. **Ama rover simülatörü** —
  itkili iniş, iniş takımı teması, propulsive descent fiziği yok. Bu repo zaten OmniLRS'nin
  materyal asset'ini (`LunarRegolith8k.mdl`) kullanıyor — yani doğrudan ilişkili/komşu çalışma.
- **Ay-iniş + RL (hazard avoidance) literatürü**: *Safe lunar landing via images* (meta-RL),
  *Meta-RL guidance/navigation/control for lunar landing* (Springer 2025) — bunlar
  **Blender/Unreal ray-tracer** ile görüntü/DEM tabanlı, Isaac Sim'in fizik motorunu
  (rijit-cisim temas simülasyonu) kullanmıyor.
- **Roket iniş RL (ay-spesifik terrain'siz)**: *Propulsive landing of launchers' first stages
  with Deep RL* (ScienceDirect), *Rocket Landing Control with Random Annealing Jump Start RL*
  (arXiv 2407.15083), *Deep RL for Six-DOF Planetary Powered Descent* (arXiv 1810.08719) —
  6-DOF iniş dinamiği var ama düz zemin/basit terrain varsayımıyla.
- **Isaac Lab framework**: *Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal
  Robot Learning* (arXiv 2511.04831) — genel framework tanıtımı, belirli bir ay inişi ortamı
  sunmuyor.

**Konumlandırma cümlesi (makalenin "related work" sonunda kullanılabilir)**: Mevcut literatürde
(1) Isaac Sim/Isaac Lab'ın rijit-cisim fizik motoruyla gerçek temas simülasyonu, (2) itkili roket
iniş dinamiği (rover değil), (3) çok-çözünürlüklü/hibrit arazi geçiş yöntemlerinin nicel
karşılaştırması — üçünü birlikte ele alan bir çalışmaya rastlanmadı. Bu makale bu boşluğu,
dar ve ölçülebilir bir kapsamda (sadece terrain alt sistemi) ele alıyor.

## 7. Önerilen Makale İskeleti (8-12 sayfa hedefi)

| Bölüm | İçerik | Tahmini sayfa |
|-------|--------|---------------|
| 1. Giriş | Problem: Isaac Sim'de ay iniş RL ortamı, temas fiziği + VRAM dengesi; sistemin genel çerçevesi (1 paragraf, tüm sistemi anlatmadan) | 1 |
| 2. İlgili Çalışmalar | §6'daki liste, konumlandırma cümlesi | 1-1.5 |
| 3. Yöntem | §3-4: mevcut mimari, üç kenar-geçiş fonksiyonu (formüllerle), hibrit tasarımı | 2-2.5 |
| 4. Deney Kurulumu | §5: preset'ler, seed'ler, metrikler, donanım | 1 |
| 5. Sonuçlar | Tablolar (§5 metrikleri), en az 1-2 grafik (örn. mask fonksiyonlarının şekli, VRAM/süre karşılaştırması) | 2-3 |
| 6. Tartışma | Hangi yöntem ne zaman tercih edilmeli; hibrit'in kazanımı; sınırlamalar (RL sonuçları henüz yok, sadece terrain alt sistemi) | 1.5-2 |
| 7. Sonuç ve Gelecek Çalışma | Kısa özet + RL eğitimi tamamlanınca genişletme planı | 0.5-1 |
| **Toplam** | | **~9-11 sayfa** |

## 8. Uygulama Checklist'i (sonraki Claude Code oturumu için)

- [ ] `_local_detail_mask()`'i `edge_fade_function` parametresine göre üç dallı hale getir
      (`smoothstep` / `gaussian` / `hybrid`), §4.2-4.4'teki formüllere göre
- [ ] `configs/terrain_config.yaml`'a `edge_fade_function`, `hybrid_inner_radius_m` alanlarını ekle
- [ ] `benchmark_terrain_quality.py`'yi §5'teki deney matrisini (4 preset × 3 yöntem × 5 seed)
      koşacak ve JSON'a yazacak şekilde genişlet
- [ ] Kenar sürekliliği (Laplacian normu) ve truncation kalıntısı metriklerini hesaplayan
      küçük yardımcı fonksiyonlar yaz (`app/terrain_generator.py` veya ayrı bir
      `app/terrain_metrics.py`)
- [ ] (Opsiyonel, Isaac Sim/GPU erişimi varsa) iniş takımı batma derinliği metriğini ekle
- [ ] Sonuçları `outputs/terrain_quality_benchmark/` altına JSON + özet tablo olarak kaydet,
      ardından basit bir grafik (matplotlib, mask fonksiyonlarının şekli + metrik karşılaştırması)
      üret
- [ ] `TWO_TIER_LANDING.md` ve `TERRAIN_QUALITY.md`'yi gerçek (kod ile tutarlı) davranışı
      yansıtacak şekilde güncelle; Yöntem B'nin dead-code durumunu ve neden terk edildiğini
      (§3.2) belgeye ekle
- [ ] Sonuç tablolarından makalenin §5 (Sonuçlar) bölümünü taslak olarak yaz

## 9. Zaman Planı (öneri)

| Gün | İş |
|-----|----|
| Gün 1 | `edge_fade_function` parametrelendirmesi + config + smoothstep/gaussian dalları (§4.2-4.3) |
| Gün 2 | Metrik hesaplama fonksiyonları (§5) + benchmark script genişletmesi, ilk koşular |
| Gün 3 | Hibrit yöntem (§4.4) implementasyonu + tam deney matrisinin koşulması |
| Gün 4 | Sonuçların analizi, grafik/tablo üretimi | 
| Gün 5-6 | Makale yazımı (§7 iskeletine göre) |
| Gün 7 | Gözden geçirme, TWO_TIER_LANDING.md/TERRAIN_QUALITY.md güncellemesi, son kontrol |

## 10. Açık Sorular / Grupla Netleştirilmesi Gerekenler

- Isaac Sim/GPU erişimi Metrik 6 (temas doğruluğu) için yeterli mi, yoksa sadece
  CPU-ölçülebilir metriklerle (1-5, 7) mi sınırlı kalınmalı?
- Hedef konferans/dergi TR dizin mi yoksa uluslararası bir workshop mu — şablon ve dil buna göre
  netleşecek.
- `r_inner` (hibrit iç yarıçap) değeri için gerçek iniş takımı geometrisinden mi (varsa CAD/USD
  asset'ten) yoksa makul bir varsayımdan mı (`assets/rocket/lunar_rocket_single_body.usda`
  kontrol edilmeli) yola çıkılacak?
