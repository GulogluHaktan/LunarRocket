# Literatür Taraması: Arazi Detay-Geçiş / LOD Karıştırma Yöntemleri

Bu tarama, `README.md` ve `results/dashboard.html`'deki deneysel bulguları bir bildirinin
"İlgili Çalışmalar" (Related Work) bölümüne yerleştirmek için yapılmıştır. Hedefli web
araması ile (ARS `deep-research` — `lit-review` modu, `bibliography_agent` +
`source_verification_agent` şablonları izlenerek) 5 tematik kümede toplam 14 doğrulanmış
kaynak derlendi. **Kapsam notu**: bu, PRISMA-uyumlu sistematik bir tarama değil, hedefli
(targeted/scoping) bir taramadır — her kümede alanın en çok atıf alan/temel referansları
önceliklendirildi, kapsamlı değildir. Her kaynak, arama sonucunda gelen sayfa (DOI, ACM/IEEE
Xplore, ScienceDirect, arXiv vb.) üzerinden tek tek doğrulandı; DOI'si arama sonucunda net
şekilde görünmeyen kaynaklarda DOI **uydurulmadı**, bunun yerine doğrulanmış URL kullanıldı.

## Küme 1 — Prosedürel terrain'de çoklu-çözünürlük blending ve LOD geçişi

**Losasso, F., & Hoppe, H. (2004).** Geometry clipmaps: Terrain rendering using nested
regular grids. *ACM Transactions on Graphics, 23*(3), 769–776.
https://doi.org/10.1145/1015706.1015799

**Duchaineau, M., Wolinsky, M., Sigeti, D. E., Miller, M. C., Aldrich, C., &
Mineev-Weinstein, M. B. (1997).** ROAMing terrain: Real-time optimally adapting meshes.
In *Proceedings of the 8th IEEE Conference on Visualization (VIS '97)* (pp. 81–88). IEEE.
https://ieeexplore.ieee.org/document/663860/

**de Boer, W. H. (2000).** *Fast terrain rendering using geometrical mipmapping*
(E-mersion Project tech. report). https://www.flipcode.com/archives/article_geomipmaps.pdf
*(gray literature / teknik rapor — akademik değil ama alanda yaygın atıf alan temel bir
teknik; Tier 3 kaynak olarak işaretlenmiştir.)*

> **Bağlantı**: Bu üç çalışma, tam olarak bizim "iki-mesh" (global + local patch) çerçevemizin
> ait olduğu geleneksel bilgisayar-grafikleri hattı — farklı çözünürlükte grid'leri tek bir
> tutarlı yüzeyde birleştirme problemi. Hepsi, **düşük ve yüksek çözünürlüklü ağların
> birleştiği sınırda "crack"/"seam" oluşmasını** merkezi bir mühendislik problemi olarak ele
> alır (ROAM'ın bintree "T-vertex" birleştirmesi, geoclipmap'in nested-grid seviyeleri arası
> geçiş bandı, geomipmapping'in blok kenarlarında "skirt"/vertex-kayması çözümleri). Bizim
> bulgumuz — as-shipped Gaussian'ın sonlu-yarıçap kesildiğinde gerçek bir dikiş bırakması —
> bu literatürün 25+ yıldır çözmeye çalıştığı **aynı sınıf problem**in somut, ölçülmüş bir
> örneğidir; smoothstep'in kompakt desteği ise ROAM/geoclipmap'in kullandığı "sınırda tam
> sıfıra inen ağırlık fonksiyonu" ilkesiyle birebir örtüşür.

## Küme 2 — Kompakt-destekli vs. sonsuz-destekli karıştırma fonksiyonlarının süreklilik özellikleri

**Wendland, H. (1995).** Piecewise polynomial, positive definite and compactly supported
radial basis functions of minimal degree. *Advances in Computational Mathematics, 4*(1),
389–396. https://doi.org/10.1007/BF02123482

**Hardy, R. L. (1971).** Multiquadric equations of topography and other irregular surfaces.
*Journal of Geophysical Research, 76*(8), 1905–1915.
https://doi.org/10.1029/JB076i008p01905

**Ohtake, Y., Belyaev, A., & Seidel, H.-P. (2003).** A multi-scale approach to 3D scattered
data interpolation with compactly supported basis functions. In *Proceedings of the 2003
Shape Modeling International (SMI '03)* (pp. 153–161). IEEE.
https://ieeexplore.ieee.org/document/1199611/

**Ebert, D. S., Musgrave, F. K., Peachey, D., Perlin, K., Worley, S., Mark, W. R., & Hart,
J. C. (2002).** *Texturing and modeling: A procedural approach* (3rd ed., s. 26–27
[smoothstep]). Morgan Kaufmann.

> **Bağlantı**: Bu küme, deneyimizin "kök neden" analizini doğrudan matematiksel literatüre
> bağlar. Hardy'nin (1971) multiquadric/Gaussian-tipi radyal fonksiyonu — bizim
> `gaussian_shipped`'imizin ailesinden — **sonsuz destekli**dir; hiçbir sonlu yarıçapta tam
> sıfıra inmez, tıpkı bizim ölçtüğümüz `exp(-1)≈%37` kalıntı gibi. Wendland (1995) ise tam
> olarak bunun çözümünü formalize eder: verilen düzgünlük (C<sup>k</sup>) için **kompakt
> destekli, minimal dereceli** pozitif-tanımlı radyal fonksiyonlar — smoothstep'in
> (kübik Hermite, Ebert ve ark., 2002) 1D, düşük-dereceli özel bir örneği olduğu ailenin tam
> matematiksel genellemesi. Ohtake ve ark. (2003), bizim hibrit yaklaşımımızın tam olarak
> izlediği stratejiyi tanımlar: **kompakt destekli baz fonksiyonlarla çok-ölçekli
> (coarse-to-fine) hiyerarşi** kurup, farklı ölçeklerde farklı karıştırma davranışı elde etmek.

## Küme 3 — Gezegen/ay inişi simülasyonlarında arazi modelleme ve temas dinamiği

**Johnson, A. E., & Montgomery, J. F. (2008).** Overview of terrain relative navigation
approaches for precise lunar landing. In *2008 IEEE Aerospace Conference* (pp. 1–10). IEEE.
https://doi.org/10.1109/AERO.2008.4526302

**NASA Jet Propulsion Laboratory.** (n.d.). *ALHAT: Autonomous Landing and Hazard Avoidance
Technology*. JPL Robotics.
https://www-robotics.jpl.nasa.gov/what-we-do/research-tasks/alhat-autonomous-landing-and-hazard-avoidance-technology/

**Barker, M. K., Mazarico, E., Neumann, G. A., Zuber, M. T., Haruyama, J., & Smith, D. E.
(2016).** A new lunar digital elevation model from the Lunar Orbiter Laser Altimeter and
SELENE Terrain Camera. *Icarus, 273*, 346–355. https://doi.org/10.1016/j.icarus.2015.07.039

**Ishida, T., Fukuda, S., Kariya, K., Kamata, H., Takadama, K., Kojima, H., Sawai, S., &
Sakai, S. (2025).** Vision-based navigation and obstacle detection flight results in SLIM
lunar landing. *Acta Astronautica, 226*, 772–781. https://doi.org/10.1016/j.actaastro.2024.11.002
*(Güncelleme: bildiri yazımı sırasında Küme 3'ü güncel bir kaynakla güçlendirmek için eklendi
— JAXA'nın Ocak 2024 SLIM görevinin uçuş verisi, arazi doğruluğunun gerçek iniş güvenliğine
etkisini somutlaştırıyor.)*

> **Bağlantı**: ALHAT programının hedefi (100 m içinde hassas iniş, LIDAR-tabanlı arazi-göreli
> navigasyon) ve LOLA/SELENE tabanlı DEM'lerin (~60 m yatay, birkaç m dikey çözünürlük)
> gösterdiği gibi, gerçek ay inişi mühendisliğinde **arazi doğruluğu doğrudan operasyonel bir
> gereksinim**dir, kozmetik bir detay değil. Bu, bizim çalışmamızın "neden dikiş/süreklilik
> önemli" motivasyonunu güçlendiriyor: ayak-temas bölgesindeki bir süreksizlik, gerçek
> ALHAT-sınıfı sistemlerde tehlike-tespiti/iniş-alanı-seçimi kararlarını etkileyebilecek türden
> bir hata sınıfına karşılık gelir — bizim deneyimiz bunu minyatür, ölçülebilir bir sayısal
> deneyle somutlaştırıyor.

## Küme 4 — Fizik-tabanlı RL ortamlarında prosedürel arazi randomizasyonu

**Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., & Abbeel, P. (2017).** Domain
randomization for transferring deep neural networks from simulation to the real world. In
*2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)* (pp.
23–30). IEEE. https://doi.org/10.1109/IROS.2017.8202133

**Rudin, N., Hoeller, D., Reist, P., & Hutter, M. (2022).** Learning to walk in minutes
using massively parallel deep reinforcement learning. In *Proceedings of the 5th Conference
on Robot Learning (CoRL 2022)*, PMLR 164, 91–100.
https://proceedings.mlr.press/v164/rudin22a.html

**Mittal, M., Yu, C., Yu, Q., Liu, J., Rudin, N., Hoeller, D., Yuan, J. L., Singh, R.,
Guo, Y., Mazhar, H., Mandlekar, A., Babich, B., State, G., Hutter, M., & Garg, A. (2023).**
Orbit: A unified simulation framework for interactive robot learning environments (Isaac
Lab'in öncülü). *IEEE Robotics and Automation Letters, 8*(6), 3740–3747.
https://doi.org/10.1109/LRA.2023.3270034

**NVIDIA. (2025).** *Isaac Lab: A GPU-accelerated simulation framework for multi-modal robot
learning*. arXiv:2511.04831. https://arxiv.org/abs/2511.04831

> **Bağlantı**: Rudin ve ark. (2022) — IsaacGym üzerinde, tam olarak bizim projemizin
> `app/terrain_pool.py`'sine benzer bir **oyun-esinli terrain-curriculum** kullanır (zorluk,
> ajanın başarı oranına göre kademeli artıyor); bu, bizim `curriculum_full_gimbal_difficulty`
> ve terrain-pool tasarımımızın izole bir örnek değil, alanın yerleşik bir pratiği olduğunu
> gösteriyor. Mittal ve ark. (2023, Orbit) ve NVIDIA (2025, Isaac Lab) doğrudan bizim
> platformumuzun kendisi — GPU-paralel, PhysX-tabanlı arazi randomizasyonunun mimari
> temelini tanımlıyorlar. Tobin ve ark. (2017), prosedürel randomizasyonun sim-to-real
> transferindeki genel motivasyonunu sağlıyor. **Ancak bu dört kaynağın hiçbiri, arazi
> randomizasyonu içindeki çoklu-çözünürlük blend *fonksiyonunun kendisinin* (smoothstep vs.
> Gaussian vs. hibrit) seçimini bir tasarım değişkeni olarak ele almıyor** — hepsi terrain
> *çeşitliliğine* (slope/rough/step vb. tip çeşitliliği) odaklanıyor, blend-şeklinin süreklilik
> maliyetine değil. Bu, çalışmamızın konumlandığı somut boşluk.

## Küme 5 — Hibrit/bölgesel karıştırma fonksiyonu birleştirme (partition of unity)

Küme 2'deki **Ohtake ve ark. (2003)** burada da merkezi kaynaktır: kompakt-destekli baz
fonksiyonların **çok-ölçekli/partition-of-unity tarzı** birleştirilmesi doğrudan bizim
hibrit yöntemimizin (temas bölgesinde bir fonksiyon, ötesinde başka bir fonksiyon, aralarında
yumuşak switch) genel matematiksel çerçevesidir.

> **Bağlantı**: Bizim `switch_weight()` fonksiyonumuz (kendisi de bir smoothstep, iki farklı
> fade fonksiyonunu ağırlıklandırarak birleştiriyor) klasik **partition-of-unity** fikrinin
> minyatür, 1D-radyal bir uygulamasıdır: `w(r)·f_gauss(r) + (1-w(r))·f_smoothstep(r)`,
> `w(r)+ (1-w(r)) = 1` kısıtıyla. Bu çerçeve altında, hibrit switch'imizin makine-hassasiyetinde
> yeni bir dikiş yaratmaması (deneyde doğrulandı) beklenen/literatürle tutarlı bir sonuçtur:
> iki C<sup>1</sup>-düzgün fonksiyonun C<sup>1</sup>-düzgün bir ağırlıkla dışbükey birleşimi
> yine C<sup>1</sup>-düzgündür.

## Sentez: Çalışmamızın literatürdeki konumu

1. **Küme 1** bize problem sınıfını verdi (çoklu-çözünürlük geçişinde dikiş/süreklilik), ama
   o literatür ağırlıklı olarak *görsel* (rendering) kaliteye odaklanıyor; biz aynı problemi
   **fizik-sorgulanabilir bir yükseklik alanı** (ayak teması, LiDAR) bağlamında,
   sayısal olarak (C0/C1 metrikleriyle) ölçtük.
2. **Küme 2**, bizim ampirik bulgumuzun ("Gaussian sonlu yarıçapta tam sıfıra inmez") *neden*
   doğru olduğunu 50+ yıllık bir matematiksel sonuçla (Hardy, 1971; Wendland, 1995) temellendiriyor
   — bu bir kodlama hatası değil, radyal baz fonksiyonlarının bilinen bir yapısal özelliği.
3. **Küme 3**, "neden önemli" sorusuna gerçek-dünya mühendislik gerekçesi sağlıyor (ALHAT,
   LOLA DEM doğruluk gereksinimleri).
4. **Küme 4**, bizim deney platformumuzun (Isaac Lab/terrain-pool) literatürdeki yerini
   gösteriyor ve **doldurduğumuz boşluğu** netleştiriyor: terrain-randomization literatürü
   *çeşitlilik*i optimize ediyor, *blend-fonksiyonu seçimini* değil.
5. **Küme 5**, önerdiğimiz hibrit yöntemi izole bir mühendislik hilesi olmaktan çıkarıp,
   yerleşik bir matematiksel çerçeveye (partition of unity / çok-ölçekli CSRBF birleştirme)
   oturtuyor.

## Sınırlamalar

- Bu, sistematik (PRISMA) bir tarama değil; hedefli bir kapsam taramasıdır (5 küme,
  küme başına 3–4 kaynak, en yüksek-etkili/temel referanslar önceliklendirildi).
- Arama İngilizce sorgularla, genel akademik web araması (WebSearch) üzerinden yapıldı;
  IEEE Xplore/ACM DL/Web of Science gibi ücretli veritabanlarına doğrudan erişim yok —
  bulunan kaynaklar bu veritabanlarının herkese açık sayfalarına (DOI, arXiv, ResearchGate
  özet sayfaları) dayanıyor.
- `de Boer (2000)` akademik hakemli bir kaynak değil (Tier 3, gray literature) ama
  geomipmapping tekniğinin birincil/orijinal kaynağı olduğu için dahil edildi.
- **Güncelleme (bildiri yazımı sırasında yapılan ikinci doğrulama turu):** Küme 3'teki iki
  kaynağın (Johnson & Montgomery, 2008; Barker ve ark., 2016) DOI'leri ilk taramada
  bulunamamıştı; bildirinin atıflarını doğrulama aşamasında yapılan hedefli aramada her
  ikisi de bulundu ve yukarıya eklendi (sırasıyla `10.1109/AERO.2008.4526302` ve
  `10.1016/j.icarus.2015.07.039`). Aynı turda Mittal ve ark. (2023)'ün yalnızca arXiv
  önbaskısı olarak listelendiği, oysa çalışmanın *IEEE Robotics and Automation Letters*'da
  hakemli olarak yayımlandığı (`10.1109/LRA.2023.3270034`) ve Ebert ve ark. (2002)'nin yazar
  listesinde iki yazarın (W. R. Mark, J. C. Hart) eksik olduğu tespit edilip düzeltildi.
