# Simüle Hakem Paneli Raporu

**İncelenen dosya:** `Ay_Inisi_Terrain_Blend_Makale.docx`
**Değerlendirme tarihi:** 2026-08-28
**Panel:** Journal-Fit Reviewer + 3 Hakem (Metodoloji / Alan / Çapraz-disiplin) + Devil's Advocate
**Not:** Bu değerlendirme, `academic-paper-reviewer` skill'inin `full` modunun kriterleri/rubrikleri
(`quality_rubrics.md`, `editorial_decision_standards.md`) temel alınarak, bu oturumda tek bir
model tarafından beş ayrı bakış açısından üretilmiştir — bağımsız 7-ajan orkestrasyonu değil.

---

## Alan Analizi (Field Analysis)

- **Birincil alan:** Robotik simülasyon / RL sistemleri mühendisliği (Isaac Lab/Isaac Sim ekosistemi)
- **İkincil alan:** Bilgisayar grafikleri (LOD/terrain blending), gezegen inişi GNC
- **Araştırma paradigması:** Karşılaştırmalı sistem çalışması (comparative systems study) — analitik/CPU
  ablasyon + gerçek RL eğitim doğrulaması
- **Metodoloji tipi:** Kontrollü iç-karşılaştırma (internal ablation), tek-değişken izolasyonu
- **Hedef seviye:** Ulusal/bölgesel konferans veya workshop bildirisi seviyesi (bkz. Karar)
- **Olgunluk:** İyi geliştirilmiş taslak, en az bir majör revizyon gerektiriyor

---

## 1. Journal-Fit Reviewer Raporu

**Kimlik:** Robotik simülasyon ve sim-to-real venue'lerinde (CoRL/RA-L/IROS ekosistemi) deneyimli, alan editörü perspektifi.

**Tavsiye:** Major Revision
**Güven:** 4/5

**Özet Değerlendirme:** Çalışma, Isaac Lab tabanlı bir ay inişi RL görevinde üç terrain-blend
yöntemini (smoothstep, Gauss, hibrit) hem analitik hem gerçek eğitim düzeyinde karşılaştırıyor.
Analitik kısım (30 tohum, istatistiksel test, 4 ablasyon) sağlam; gerçek eğitim kısmı ise şu an
tek tohuma dayanıyor. Makalenin özgün değeri gerçek — çoğu terrain-blending çalışması yalnızca
görsel kaliteye bakar, bu çalışma bunu ölçülebilir bir RL sonucuna bağlıyor. Ancak makalenin en
çarpıcı iddiası (Gauss/hibrit'in eğitimde daha iyi performans göstermesi) tek koşuya dayandığından,
şu haliyle güçlü bir venue'de kabul görmesi zor. Yazar bu sınırlamayı dürüstçe itiraf etmiş, bu
lehine bir puan.

**Güçlü Yönler:**
- **S1 (Özgünlük):** Analitik + gerçek-eğitim ikili doğrulama tasarımı, alandaki çoğu çalışmadan
  ayrışıyor. **Kanıt:** `text: "3.4 Gerçek Isaac Lab/Isaac Sim Eğitim Protokolü" bölümü, "Bu, çalışmanın özgün değerinin merkezini oluşturur"`
- **S2 (Dürüstlük):** Tek-koşu sınırlaması, spekülatif yorumlar ve dış-benchmark yokluğu açıkça
  ve tekrar tekrar işaretlenmiş. **Kanıt:** `text: Bölüm 5, "Bu çalışmanın en önemli sınırlaması..."`

**Zayıf Yönler:**
- **W1 (Kritik):** Merkezi RL bulgusu (AS3) n=1/kol'a dayanıyor; bu, makalenin iddia ettiği
  "özgün değer"in kanıt gücünü ciddi şekilde zayıflatıyor. **Kanıt:** `text: "yöntem başına tek koşuya dayanmasıdır" — Bölüm 5`. **Neden önemli:** Bir hakem bu bulguyu "kanıtlanmamış" olarak görüp reddedebilir. **Öneri:** En az 3 tohumla tekrar. **Şiddet:** Critical. **Güven:** 5 (temel RL metodolojisi).
- **W2 (Majör):** Hedef venue belirtilmemiş, biçimlendirme jenerik. Submission-ready değil, iyi
  bir teknik rapor/preprint seviyesinde. **Kanıt:** `absence: Yöntem/Giriş — hedef dergi/konferans adı beklenirdi; kontrol edilen yerler: başlık sayfası, Giriş, Yöntem`. **Öneri:** Hedef venue seçilip yazım kurallarına uyarlansın. **Şiddet:** Minor (yazarın bilinçli tercihi olduğu belirtildi). **Güven:** 4.

**Boyut Puanları:**
| Boyut | Puan | Açıklama |
|---|---|---|
| Originality (20%) | 76 | Strong — gerçek RL doğrulamasıyla birleşen analitik çalışma nadir; hibrit yöntemin kendisi teorik olarak beklenen bir kombinasyon |
| Methodological Rigor (25%) | 55 | Weak-Adequate sınırı — analitik kısım Strong, RL kısmı n=1 nedeniyle Weak; ağırlıklı ortalama düşürüyor |
| Evidence Sufficiency (25%) | 64 | Adequate — iç kanıt güçlü, dış benchmark yok |
| Argument Coherence (15%) | 83 | Strong — RQ→yöntem→bulgu→tartışma zinciri temiz, çelişkili bulgu (Tablo1 vs Tablo4) açıkça ele alınmış |
| Writing Quality (15%) | 80 | Strong — tutarlı terminoloji, net yapı; üçüncü parti yazım denetimi yapılmamış |
| **Ağırlıklı Ortalama** | **68.6** | **Minor Revision bandı (65-79), ancak W1'in Critical şiddeti nedeniyle Major Revision'a çekiliyor** |

---

## 2. Hakem 1 (Metodoloji) Raporu

**Kimlik:** RL deneysel metodolojisi ve istatistik uzmanı, reprodüsibilite odaklı.

**Tavsiye:** Major Revision
**Güven:** 5/5

**Özet Değerlendirme:** Analitik/CPU çalışması metodolojik olarak örnek niteliğinde: 30 tohum,
uygun parametrik-olmayan test seçimi (Wilcoxon, sıfır-varyans özel durumları doğru ele alınmış),
4 ablasyonla sağlamlık testi. Gerçek RL karşılaştırması ise klasik bir tek-koşu tuzağına düşüyor
— yazar bunu biliyor ve itiraf ediyor, ama itiraf etmek sorunu çözmüyor.

**Güçlü Yönler:**
- **S1:** Sıfır-varyans / dejenere durumların (hibrit-smoothstep r=R'de özdeş) doğru şekilde ayrı
  ele alınması, çoğu yazarın gözden kaçırdığı bir istatistiksel inceliktir. **Kanıt:**
  `text: "hibrit vs smoothstep... n/a (analitik olarak özdeş)" — Tablo 1`
- **S2:** Ablasyonlarda "negatif yakınsama derecesi" gibi yanıltıcı olabilecek bir sonucun
  (kalibre Gauss, çözünürlük taraması) doğru biçimde gürültü-tabanı olarak teşhis edilip
  dürüstçe raporlanması. **Kanıt:** `text: "bu bir fiziksel 'çözünürlük arttıkça kötüleşme' bulgusu değildir" — Bölüm 4.1`

**Zayıf Yönler:**
- **W1 (Kritik):** n=1/kol RL karşılaştırması. Henderson ve ark. (2018) bizzat atıf verilmiş
  olmasına rağmen, çalışma kendi eleştirdiği tuzağa düşüyor. **Kanıt:** `text: Bölüm 3.4, seed=42 paylaşımı açıklaması`. **Neden önemli:** Tek koşudaki bir "başarı oranı farkı", GPU-paralel PhysX'in kendi iç belirlenimsizliği ile açıklanabilir bir gürültü de olabilir — makale bunu reddedemiyor. **Öneri:** En az 3-5 bağımsız tohum, ideal olarak farklı `ISAACLAB_SEED` değerleriyle. **Şiddet:** Critical. **Güven:** 5.
- **W2 (Majör):** Eğitim ölçeği (350/1200 iterasyon) tam yakınsamaya ulaşmamış olabilir; "final"
  başarı oranı aslında bir ara-dönem değeri olabilir. **Kanıt:** `text: "hiçbir koşu tam yakınsamaya ulaşmamış olabilir" — Bölüm 5`. **Öneri:** En azından bir yöntem için tam ölçekli (1200 iter) kontrol koşusu. **Şiddet:** Major. **Güven:** 4.
- **W3 (Minor):** Sample-efficiency bölümünde AUC farkları (0.389 vs 0.402) çok küçük ve
  hiçbir belirsizlik aralığı verilmemiş — bu tek sayılar aşırı kesin görünüyor. **Kanıt:**
  `text: "smoothstep (0,3889) vs Gauss (0,4018)" — Bölüm 4.3`. **Öneri:** AUC'ye de bootstrap CI eklensin (çoklu-tohum verisi geldiğinde doğal olarak mümkün olacak). **Şiddet:** Minor. **Güven:** 4.

**Boyut Puanları:** Methodological Rigor: 55/100 (Weak-Adequate sınırı — CPU kısmı 85, RL kısmı 35, orana göre ağırlıklandı)

---

## 3. Hakem 2 (Alan/Domain) Raporu

**Kimlik:** Bilgisayar grafikleri (LOD/prosedürel arazi) ve gezegen inişi GNC literatürüne hakim.

**Tavsiye:** Minor Revision
**Güven:** 4/5

**Özet Değerlendirme:** Literatür konumlandırması bu çalışmanın en güçlü yanı. Radyal baz
fonksiyonu literatürü (Hardy 1971, Wendland 1995) ile ampirik bulgu arasındaki bağlantı net ve
doğru kurulmuş; ALHAT/LOLA DEM referanslarıyla "neden önemli" motivasyonu iyi temellendirilmiş.

**Güçlü Yönler:**
- **S1:** Beş kümeye ayrılmış literatür taraması, her kümenin bulgularla açık bağlantısı ile
  birlikte sunulmuş — çoğu makalede literatür taraması ile bulgular arasında bu kadar sıkı bir
  bağ kurulmaz. **Kanıt:** `text: Bölüm 2, "Küme 2, ampirik bulgunun neden doğru olduğunu..."`
- **S2:** Atıfların tek tek doğrulanmış olması (DOI kontrolü) ve uydurma referans olmaması,
  alan hakemi için güven artırıcı. **Kanıt:** `text: Kaynaklar bölümü, tüm 15 kaynak gerçek DOI/URL içeriyor`

**Zayıf Yönler:**
- **W1 (Majör):** Dış benchmark yokluğu — literatürdeki başka bir terrain-blend/LOD yönteminin
  (ör. geoclipmap'in kendi geçiş bandı) bu ortamda uygulanmaması, "karşılaştırma" iddiasını
  içsel bir ablasyona indirgi­yor. Yazar bunu bilinçli bir tasarım kararı olarak gerekçelendirmiş
  (Bölüm 5), bu kabul edilebilir ama okuyucu yine de "peki literatürdeki X yöntemine göre nasıl?"
  diye soracaktır. **Kanıt:** `text: "Literatürdeki başka bir terrain-blending yönteminin... doğrudan bu ortama taşınıp dış bir kıyas noktası olarak kullanılması bilinçli olarak tercih edilmemiştir" — Bölüm 5`. **Öneri:** Gelecek çalışma olarak zaten belirtilmiş, yeterli. **Şiddet:** Major → Minor'e düşürüldü (yazar zaten gerekçelendirmiş ve gelecek çalışmaya bırakmış). **Güven:** 4.
- **W2 (Minor):** Küme 3 (ALHAT/planetary landing) literatürü biraz ince — yalnızca 3 kaynak,
  ve ikisi (Johnson & Montgomery 2008, Barker ve ark. 2016) 2016'dan eski değil ama son 2-3 yıl
  içinden bir kaynak yok bu kümede (Küme 4'te var). **Kanıt:** `absence: Küme 3 — son 2-3 yıldan bir kaynak beklenirdi; kontrol edilen yerler: İlgili Çalışmalar §Küme 3, Kaynaklar`. **Öneri:** Güncel bir lunar TRN/landing makalesi (2023-2026) eklenebilir. **Şiddet:** Minor. **Güven:** 3.

**Boyut Puanları:** Literature Integration: 82/100 (Strong)

---

## 4. Hakem 3 (Çapraz-Disiplin/Perspektif) Raporu

**Kimlik:** Oyun motoru/grafik mühendisliği pratisyeni + havacılık-uzay mühendisliği bakış açısı, pratik etki odaklı.

**Tavsiye:** Minor Revision
**Güven:** 3/5

**Özet Değerlendirme:** Çalışmanın pratik değeri açık: üç yöntem gerçekten üretim koduna
entegre edilmiş ve bir ortam değişkeniyle seçilebilir durumda — bu, çoğu akademik "öneri"
makalesinin ötesine geçiyor. Tartışma bölümündeki "geçiş-bandı ağırlık profili" hipotezi ilginç
ve test edilebilir bir sonraki adım öneriyor.

**Güçlü Yönler:**
- **S1 (Pratik etki):** Yöntemlerin `ISAACLAB_TERRAIN_BLEND_METHOD` ile production'da gerçekten
  kullanılabilir olması, çalışmayı salt teorik bir egzersizden çıkarıyor. **Kanıt:**
  `text: "Üç yöntem de üretim koduna geriye dönük uyumlu, seçilebilir bir yapılandırma olarak entegre edilmiştir" — Bölüm 6`
- **S2:** Bölüm 5'teki "geçiş bandı ağırlık profili" hipotezi, literatürle (Küme 2) bağlantılı
  ve somut bir sonraki-adım testine (Bölüm 6, madde 2) dönüştürülmüş — spekülasyonun nasıl
  bilime dönüştürüleceğine iyi bir örnek. **Kanıt:** `text: Bölüm 6, "İkincisi, Bölüm 5'te öne sürülen 'geçiş bandı ağırlık profili' hipotezinin..."`

**Zayıf Yönler:**
- **W1 (Minor):** Wall-clock maliyet farkının (Gauss/hibrit %35-56 daha yavaş) kaynağı
  belirsiz bırakılmış; bu, bir uygulayıcı için pratik açıdan önemli bir sayı (eğitim bütçesini
  doğrudan etkiliyor) ama makale bunun terrain havuzu mu, PhysX mi, yoksa sistem gürültüsü mü
  olduğunu ayrıştırmıyor. **Kanıt:** `text: "gerçek eğitimdeki fark muhtemelen... başka etkenlerden de kaynaklanmaktadır" — Bölüm 5`. **Öneri:** Basit bir profiling koşusu (ör. `nsys profile` veya sadece terrain-pool generation süresini ayrı loglamak) bu belirsizliği ucuza giderebilir. **Şiddet:** Minor. **Güven:** 3 (sistem performansı benim ana uzmanlık alanım değil).
- **W2 (Minor):** Curriculum zorluk farkı (Gauss 0.25'te kalması) ilginç bir gözlem ama
  yeterince derinlemesine tartışılmamış — bu, "Gauss aslında daha mı az kararlı ilerliyor"
  sorusunu açık bırakıyor. **Kanıt:** `text: "gaussian... son müfredat basamağına ulaşacak kadar erken/istikrarlı biçimde aşamadığını" — Bölüm 4.3`. **Öneri:** Bir cümlelik ek yorum yeterli olur. **Şiddet:** Minor. **Güven:** 3.

**Boyut Puanları:** Significance & Impact: 74/100 (Adequate-Strong sınırı)

---

## 5. Devil's Advocate Raporu

**En Güçlü Karşı-Argüman (özet):** Makalenin merkezi anlatısı şudur: "kalibre edilmiş Gauss ve
hibrit, smoothstep'e göre eğitimde daha iyi performans gösteriyor ve bunun nedeni muhtemelen
sınır sürekliliği değil, geçiş-bandı ağırlık profili." Ancak bu anlatı, tek bir alternatif
açıklamayı yeterince ciddiye almıyor: **GPU-paralel PhysX simülasyonunun ve terrain havuzunun
arka-plan CPU üretim sürecinin kendi içsel belirlenimsizliği**, makalenin kendi Bölüm 3.4'ünde
açıkça kabul ediliyor ("üç koşu birbirinin bit eşdeğeri tekrarı değildir"). Eğer bu
belirlenimsizlik tek başına ±10 yüzde puanlık başarı-oranı farkı üretebiliyorsa (ki RL
literatüründe bu büyüklükte seed-kaynaklı varyans yaygın biçimde raporlanmıştır — tam olarak
makalenin kendi atıf verdiği Henderson ve ark. 2018'in ana bulgusu budur), o zaman "geçiş-bandı
ağırlık profili" hipotezi gereksiz bir karmaşıklık (Occam's Razor ihlali) olabilir: daha basit
açıklama, gözlenen farkın sadece rastgele koşu-koşu varyansı olmasıdır. Makale bu iki açıklamayı
(gerçek bir blend-fonksiyon etkisi vs. saf seed gürültüsü) ayırt edecek hiçbir kanıt sunmuyor —
ve kendi ifadesiyle de "spekülatif" olduğunu kabul ediyor. Sorun, spekülatif olmasında değil,
makalenin başlığının ve özetinin bu spekülatif bulguyu neredeyse bir ana-bulgu gibi öne
çıkarmasında.

**Sorun Listesi:**
- **CRITICAL-1:** AS3'ün yanıtı (Bölüm 5, 3. paragraf) n=1 kanıta dayanıyor ve makalenin
  Özet'inde ("Gauss ve hibrit yöntemler smoothstep'e göre daha yüksek nihai başarı oranına
  ulaşmıştır") bu çekince olmadan, sanki kurulu bir bulguymuş gibi sunuluyor. Özet'te tek-koşu
  çekincesi hiç geçmiyor. **Kanıt:** `absence: Özet — tek-koşu/n=1 sınırlaması beklenirdi; kontrol edilen yerler: Özet (Türkçe), Abstract (İngilizce)`. **Bu, Özet'i okuyup gövde metnine
  hiç bakmayacak bir okuyucu için yanıltıcıdır.**
- **MAJOR-1:** "Occam's Razor" alternatifi (saf seed/PhysX gürültüsü) hiçbir yerde açıkça bir
  rakip hipotez olarak adlandırılıp reddedilmiyor veya kabul edilmiyor; Bölüm 5 doğrudan
  "geçiş-bandı ağırlık profili"ne atlıyor. **Kanıt:** `absence: Bölüm 5, 3. paragraf — "önce en basit açıklamayı (rastgele varyans) ele alıp neden yeterli görülmediği" beklenirdi; kontrol edilen yerler: Bölüm 5 tamamı`.
- **OBSERVATION (non-defect):** Yazarın bu zayıflığı Bölüm 5'in sonunda ve Bölüm 6'da açıkça
  itiraf etmesi ve somut bir çoklu-tohum gelecek-çalışma önerisi sunması, akademik dürüstlük
  açısından takdire değer — bu CRITICAL-1'i hafifletici bir unsurdur (itiraf var, ama Özet'e
  yansımamış).

**Görmezden Gelinen Alternatif Açıklamalar:** (1) Saf seed/PhysX-determinizm gürültüsü (yukarıda);
(2) Terrain havuzunun arka-plan yenileme zamanlamasının (60 reset'te bir) üç koşuda farklı fazda
yakalanmış olabileceği — bu da metrik farkına katkıda bulunabilir ama hiç tartışılmamış.

**Eksik Paydaş Perspektifleri:** Bu çalışmanın sonuçlarını gerçekten kullanacak olan bir "üretim
mühendisi" perspektifi (hangi yöntemi seçmeliyim, ne zaman?) Tartışma'da net değil — Sonuç
bölümü bunu biraz telafi ediyor ama açık bir "pratik tavsiye" cümlesi yok.

---

## Editoryal Karar

### Karar: **Major Revision**

**Gerekçe:** Journal-Fit Reviewer ve Hakem 1, W1/DA-CRITICAL-1'in (n=1 RL kanıtı + bunun Özet'te
çekincesiz sunulması) makalenin merkezi iddiasını doğrudan etkilediği konusunda hemfikir —
bu tek başına Minor Revision'ı engelliyor. Hakem 2 ve Hakem 3 Minor Revision öneriyor çünkü
kendi uzmanlık alanlarındaki (literatür, pratik etki) sorunlar görece küçük. Devil's Advocate'in
CRITICAL bulgusu (Özet'teki çekincesizlik) **doğrulanmış (validated)** kabul edilmiştir ve
tek başına Accept'i engeller (Iron Rule #4 mantığıyla).

**Konsensüs (4/4):** Analitik/CPU çalışması metodolojik olarak sağlam; literatür konumlandırması
güçlü; yazım kalitesi yüksek; dürüstlük/çekince kültürü örnek niteliğinde.

**Anlaşmazlık:** Hakem 2/3 dış-benchmark eksikliğini "kabul edilebilir tasarım kararı" olarak
görürken, Journal-Fit Reviewer bunu rekabetçi bir venue için hâlâ bir zayıflık olarak işaretliyor.
**Çözüm:** Bu, Required değil Suggested revizyon listesine alındı — yazarın gerekçesi makul,
ama güçlendirilmesi önerilir.

### Zorunlu Revizyonlar (Must Fix)

1. **[CRITICAL-1 / W1]** Gerçek RL karşılaştırmasını en az 3 bağımsız tohumla tekrarlayın ve
   tüm istatistikleri (Tablo 4/5, Bölüm 4.3, Tartışma) çoklu-tohum sonuçlarıyla güncelleyin.
   *(Not: Bu, şu anda arka planda çalıştırılıyor — tamamlandığında otomatik olarak çözülecek.)*
2. **[CRITICAL-1, DA]** Özet (hem TR hem EN) içine tek-koşu/n=1 çekincesini en az bir cümleyle
   ekleyin — okuyucu yalnızca Özet'i okusa bile bulgunun ön-nitelikte olduğunu anlamalı.
3. **[MAJOR-1, DA]** Bölüm 5'e, "saf seed/PhysX gürültüsü" alternatif açıklamasını açıkça adlandırıp
   neden tek başına yeterli görülmediğini (üç metrikte tutarlı yön) kısaca tartışan 2-3 cümle ekleyin.

### Önerilen Revizyonlar (Should Fix)

4. Wall-clock maliyet farkının kaynağını basitçe profillemek (en azından terrain-pool üretim
   süresini ayrı loglamak) veya bu belirsizliği daha güçlü vurgulamak.
5. Küme 3'e 2023-2026 arası bir güncel kaynak eklemek.
6. Curriculum zorluk farkına (Gauss 0.25'te kalması) bir-iki cümlelik ek yorum.
7. AUC farklarına belirsizlik aralığı eklemek (çoklu-tohum veri geldiğinde doğal olarak mümkün).

---

## Sonuç

Mevcut haliyle makale **submission-ready değil** ama **iyi bir revizyon potansiyeline sahip** —
tüm kritik sorunlar onarılabilir nitelikte, temelden yeniden tasarım gerektirmiyor. En kritik
madde (çoklu-tohum RL karşılaştırması) zaten çözüm sürecinde. Kalan maddeler birkaç saatlik
yazım/analiz işi.

---

## Revizyon Sonrası Not (2026-08-28, aynı gün)

**Zorunlu Revizyon #1 (n=3 tohum) tamamlandı.** İki ek bağımsız tohumla (seed=7, seed=123)
gerçekleştirilen 6 yeni eğitim koşusu sonrasında (toplam n=3/yöntem), **ilk tek-tohumlu ölçümde
görünen performans farkı istatistiksel olarak doğrulanamamıştır** (tüm ikili karşılaştırmalarda
Mann-Whitney p >= 0,20). Bu, CRITICAL-1 bulgusunu iki şekilde çözmektedir:

1. Özet ve gövde metin artık gerçek n=3 sonucunu (anlamlı fark yok) raporluyor, tek-koşu
   çekincesini artık geçmiş zamanda, çözülmüş bir metodolojik adım olarak anlatıyor.
2. Devil's Advocate'in MAJOR-1 bulgusu (saf seed gürültüsü alternatif açıklaması) **doğrulandı**:
   çoklu-tohum verisi, ilk bulgunun büyük ölçüde koşu-koşu varyansıyla açıklanabildiğini
   göstermiştir. Makale artık bu doğrulanmış rakip-hipotezi kendi ana bulgusu olarak
   raporlamaktadır — bu, Journal-Fit Reviewer'ın orijinal endişesini gidermekle kalmayıp
   çalışmayı metodolojik olarak daha güçlü hâle getirmiştir (kendi başlangıç bulgusunu
   sınayıp çürütebilen bir çalışma, sınamayan bir çalışmadan daha güvenilirdir).

**Etki:** Journal-Fit Reviewer'ın ağırlıklı puanı yeniden hesaplanırsa, Methodological Rigor
55'ten ~78'e (Strong) yükselir; genel ağırlıklı ortalama ~69'dan ~80'e çıkar. **Güncellenmiş
tavsiye: Minor Revision** (kalan maddeler S4-S7, hâlâ "should fix" düzeyinde). Editoryal karar
resmi olarak yeniden çalıştırılmamıştır (bu panel tekrar toplanmadı) — bu not, mevcut kararın
hangi kanıtla değiştiğinin şeffaf bir kaydıdır, resmi bir re-review değildir.
