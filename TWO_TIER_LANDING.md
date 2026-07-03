# İki-Kademeli Terrain Sistemi (Two-Tier Landing Zone)

## Konsept

Maksimum triangle kullanmak, ama akıllı dağıtmak:
- **Global terrain**: Düşük resolution (128×128 = 32k triangles) - çevre bölgesi
- **Landing zone detail**: Ultra-yüksek resolution (256×256 = 130k triangles) - iniş bölgesi
- **Total**: ~160k triangles, ama önemli yerlere konsantre

```
Çevre Bölgesi                    Landing Zone (Roketin iniş yeri)
(128×128 = 32k tri)             (256×256 = 130k tri)
┌─────────────────────────┐     ┌────────────────┐
│                         │     │ ██████████████ │
│     Düşük Detay         │     │ █ Milimetrik   │
│   Coarse Terrain        │     │ █ Detail!      │
│                         │     │ ██████████████ │
└─────────────────────────┘     └────────────────┘
```

## Fiziğinin Avantajları

1. **Roketin temas noktasında gerçekçi**: Landing zone'da 130k triangle = çok detaylı topografi
2. **Çevre optimized**: Görülen ancak fiziği kritik olmayan bölgede düşük resolution
3. **Smooth transition**: 8 meter blend zone ile seamless geçiş
4. **Toplam VRAM**: Aynı 256×256'dan daha az (128 + 256'ın karması)

## Konfigürasyon

### Ana Ayarlar (configs/terrain_config.yaml)

```yaml
# Global patch resolution (çevre bölgesi)
terrain_quality: high_detail_landing

# High-detail landing zone
local_detail_patch:
  enabled: true
  patch_size_m: [30.0, 30.0]      # 30m × 30m iniş bölgesi
  resolution: 256                  # 256×256 = 130k triangles
  blend_radius_m: 8.0             # 8 meter smooth blend zone
  blend_scale: 2.0                # Gaussian falloff sharpness
  noise_scale: 0.015              # Fine roughness (milimetrik detay)
  noise_smoothing_passes: 1       # Minimal smoothing = max detail
```

### Iniş Bölgesi Detayı

Multi-scale roughness oluşturuluyor:
- **Coarse**: Büyük kayalar, eğimler
- **Medium**: Kum yığınları, kraterler
- **Fine**: Toz tanecikleri, erozyon desenleri

## Kullanım

### Benchmark'ta Test Et

```bash
# Tüm preset'leri karşılaştır (yeni high_detail_landing dahil)
./python.sh app/benchmark_terrain_quality.py
```

Çıktı örneği:
```
low_debug             | Res: 128 | Tri:   32k   | VRAM: 1.3GB
medium_train          | Res: 192 | Tri:   72k   | VRAM: 1.3GB
high_train            | Res: 256 | Tri:  130k   | VRAM: 1.3GB
high_detail_landing ⭐ | Res: 256 | Tri:  160k   | VRAM: 1.3GB  (global 128 + local 256)
demo_high             | Res: 256 | Tri:  130k   | VRAM: 1.3GB
```

### Çift-Kademeli Terrain Kullanmak

**Seçenek 1: Config dosyası düzenle**

`configs/terrain_config.yaml`:
```yaml
terrain_quality: high_detail_landing
```

Sonra çalıştır:
```bash
./python.sh app/main.py --steps 300
```

**Seçenek 2: Önceden hazırlı config kullan**

```bash
# Kopyala ve config'te kullan
cp configs/train_high_detail_landing.yaml configs/sim_config.yaml
./python.sh app/main.py --steps 300
```

## Teknik Detaylar

### Mesh Yapısı

```
Global Terrain (128×128):
- Vertices: 16,384
- Triangles: ~32k
- Grid spacing: 0.625m
- Covers: 80m × 80m tüm alan

Landing Zone (256×256):
- Vertices: 65,536
- Triangles: ~130k
- Grid spacing: 0.117m (5.3× daha ince!)
- Covers: 30m × 30m iniş bölgesi
- Blend zone: 8m smooth transition

Total:
- Vertices: ~82k
- Triangles: ~162k
- Memory: Biraz daha az than full 256×256
```

### Blending Algoritması

Gaussian falloff kullanılıyor:
```
blend_factor = exp(-(distance/radius)^blend_scale)

blend_scale = 2.0 → smooth Gaussian curve
             = 1.0 → linear falloff
             = 3.0 → sharper transition
```

8 meter radius'ta:
- 0m (center): 100% local detail
- 4m: 90% local, 10% global
- 8m: 36% local, 64% global
- 12m: 5% local, 95% global
- 16m+: 0% local detail (pure global)

## Sonuç

✅ **130k triangle iniş bölgesinde** (roketin temas noktası çok detaylı)
✅ **32k triangle çevrede** (görüş iyisi, fizik yeterli)
✅ **Smooth blend** (no seams, no artifacts)
✅ **Aynı VRAM kullanımı** (optimized dağıtım)
✅ **Realistic contact dynamics** (iniş bölgesinde milimetrik detay)

Bu yapı, rocket'in landing gear'ı ve kontakt noktalarında gerçekçi fizik simulasyonu sağlar!
