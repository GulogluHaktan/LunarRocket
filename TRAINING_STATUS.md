# LunarRocket SAC eğitimi — durum raporu (2026-09-17)

**GOAL DURDURULDU** (kullanıcı talebiyle). Bu dosya, bu oturumda yapılan ~50M+ adımlık
eğitim/reward-mühendisliği çalışmasının özetidir; temiz context ile devam etmek için okunmalı.

## En önemli, hemen aksiyon gerektiren bulgu

Son tur (`curriculum_initial_difficulty` 0.05 → 0.0, checkpoint
`logs/sb3_sac/LunarRocket-Lander-Direct-v0/2026-09-17_17-42-20/model.zip`) **hâlâ dik
yükselirken bitti**:

- 5 eşit dilimde success_rate: %0.2 → %0.4 → %0.2 → **%9.3 → %34.8**
- son-1/3 ortalama: **%26.4**, en yüksek pencere **%76.9**
- gate'ler son-1/3'te %59-76 (önceki 4 başarısız denemede %6-20'ye çökmüştü)
- `Curriculum/difficulty` de ilk kez sabit 0.05'te kalmayıp kıpırdadı (max 0.10)

Kullanıcının kendi kuralı ("başarı 5M sonunda hâlâ artarken bittiyse sadece devam ediyoruz")
bu turun tam olarak devam ettirilmesi gereken durum. **Bir sonraki adım**: bu checkpoint'ten
(sıfırdan değil, resume ile) devam ettirip trendin gerçekten platoya mı oturduğunu yoksa
gerçekten %50'yi aşıp aşmadığını görmek.

## Mevcut config durumu (baseline, kanıtlanmış — env_cfg.py)

- `rew_timeout = -250.0` (rew_crash=-220'den kötü — "hiç deneme" cezası)
- `soft_vertical_speed_mps = 0.6` (whiteboard 0.5'ten gevşetildi, ama 0.75'ten geri çekildi —
  crash sınırına (1.0) tampon açmak için)
- `soft_horizontal_speed_mps = 0.7` (whiteboard 0.5'ten, kullanıcı onaylı sapma)
- `target_radius_m = 2.0` (whiteboard'da yok, 1.0'dan gevşetildi)
- `rew_wb_vz_weight = 3.0` (2.0'dan artırıldı, dikey hız kontrolü için)
- `rew_wb_loiter_penalty_per_s = 10.0` (5.0'a düşürme denendi, BAŞARISIZ — geri alındı)
- `rew_wb_readiness_weight = 2.5` (8.0'dan düşürüldü — "bedava hoverlama" bug'ı yüzünden)
- `slammed` (çok hızlı iniş) artık `harsh` gibi kısmi kalite kredisi alıyor (önceden `left_bounds`
  ile aynı düz -220 cezasıydı)
- `policy_min_gimbal_deg = 4.0`, `policy_max_gimbal_deg = 8.0` (10/15 ve 6/10 denendi, İKİSİ DE
  BAŞARISIZ — geri alındı, ayrıntı aşağıda)
- **`curriculum_initial_difficulty = 0.0`** (0.05'ten düşürüldü — bu turun tek değişkeni, ve tek
  başarılı/umut verici sonuç bu oldu)

## Bu oturumda denenen ve BAŞARISIZ olan 4 ardışık müdahale (hepsi tek değişken, hepsi
gate'lerin neredeyse tamamını çökertti, sadece hedeflenen ekseni değil)

1. **Gimbal floor 4°→10° / ceiling 8°→15°**: whole=%1.2, tüm gate'ler %17-36'ya düştü
   (sadece gate_vertical %58-68'de tutundu). Teori: 4° yetkisiyle 1 m/s yatay hatayı düzeltmek
   ~9s sürüyor (g·tan(4°)≈0.11 m/s²), bu yüzden daha fazla yetki iyi gelir sanıldı — ama SAC'ın
   keşif gürültüsü eylem aralığıyla büyüdüğü için kendi kendini kararsızlaştırdı.
2. **Gimbal floor 4°→6° / ceiling 8°→10°** (daha küçük adım): whole=%0.5, aynı çöküş paterni.
   Sıçramanın büyüklüğü değil, gimbal tabanını **her türlü** yükseltmek sorunlu.
3. **Loiter grace period (9s cezasız)**: whole=%0.0 (484 pencerenin TAMAMI sıfır). Politika
   gerçekten daha sabırlı oldu (mean_episode_length 15.9-18.5s'e çıktı) ama bu iyi inişe
   dönüşmedi, sadece aynı kötü sonucu geciktirdi.
4. **Loiter 10.0→5.0** (magnitude azaltma): whole=%0.4, aynı çöküş. Teori (erken çakılmanın
   matematiksel olarak daha ucuz olması, "rush-to-fail") muhtemelen hâlâ doğru ama bu şekilde
   test edilmesi sistemi bozdu.

**Sonuç**: `{loiter=10.0, gimbal=4°/8°, timeout=-250}` etrafındaki denge kırılgan bir optimum —
etrafındaki hiçbir tek-parametre değişikliği zarif bozulma göstermiyor, hepsi feci çöküşe
yol açıyor. Curriculum zorluğu (görev kolaylığı) değişikliği bu paterni KIRDI ve iyi sonuç verdi
— muhtemelen reward-timing değişkenleri yerine görev-zorluğu ekseninin daha güvenli/verimli bir
kaldıraç olduğunu gösteriyor.

## Reward kompozisyonu bulgusu (en iyi eski referans turdan, %27 whole-run)

Episode başına ortalama: `loiter` ≈ -170, `terminal` ≈ -227 (toplam ≈ -400) vs. pozitif
shaping (velocity+proximity+tilt+rel_xy+readiness) ≈ +100. Yani reward'ın ~4 katı ağırlığı
"acele et/çakma" baskısında, hassasiyet sinyali görece küçük. Bu, tek tek gate'lerin (%70-85)
6'sını AYNI ANDA tutturma (joint constraint) sorununu açıklıyor.

## Yapısal bulgular (kod incelemesi)

- `curriculum_success_threshold=0.35` (512-episode kayan pencere) bu oturum boyunca hemen hemen
  hiç aşılmadı → curriculum zorluğu + gimbal yetkisi neredeyse hep tabanda kaldı.
- Terrain eğimi hafif (`terrain_slope_scale=0.035` → maks ~2.8°), yani zemin eğimi başlı başına
  darboğaz değil.
- `landed` tetikleyicisi: en yakın ayağın zemine 4cm mesafesi — sensör/eşik sorunlu değil,
  davranışsal (roket o kadar yaklaşmayı seçmiyor/seçemiyor).

## GOAL DURDURULDU — kullanıcı `/goal stop` çalıştırdı (bu bölümden sonra yazıldı)

Kullanıcı "son bir kez bunu devam ettir, sonra clear at" dedi; `2026-09-17_17-42-20/model.zip`'ten
bir devam turu daha yapıldı (checkpoint: `2026-09-17_18-32-01/model.zip`). Sonuç:

- whole-run success %24.6, son-1/3 %17.0, en yüksek pencere %72.7
- 5 dilimde success: %46.9 → %26.4 → %14.8 → %19.3 → %15.8 (düşüş eğiliminde)
- **AMA `Curriculum/difficulty` bu oturumda İLK KEZ gerçekten ilerledi**: whole=0.217,
  son-1/3=0.187, max=0.30 (önceki HER turda 0.05'te sabit kalmıştı). Yani success'teki düşüş
  muhtemelen beceri kaybı değil, curriculum ilerledikçe görevin (spawn mesafesi + gimbal yetkisi
  birlikte) zorlaşması. gate'ler son-1/3'te hâlâ makul (%51-68).

## Önerilen bir sonraki adım (yeni context için)

1. Bu checkpoint (`2026-09-17_18-32-01/model.zip`) curriculum'un gerçekten ilerlediği TEK
   checkpoint — sonraki turda bundan devam edip curriculum zorluğunun daha da ilerleyip
   ilerlemediğini ve success'in zorlanan görevde nasıl davrandığını izlemek en bilgilendirici
   olur.
2. `curriculum_initial_difficulty=0.0` değişikliği (önceki turun tek değişkeni) kanıtlanmış en
   güvenli/verimli kaldıraç — reward-timing değişkenlerinden (loiter/gimbal/timeout — 4/4
   başarısız) kaçınıp bu eksende devam edilmeli.
3. Eğer curriculum ilerlemesi devam ederse, `Curriculum/difficulty` yükselirken success'in nasıl
   davrandığını (düşüyor mu, yoksa yeni zorlukla birlikte toparlanıyor mu) izlemek asıl soruyu
   cevaplayacak: mevcut politika sadece kolay görevi mi ezberledi, yoksa genellenebilir bir
   beceri mi öğrendi.
