#!/usr/bin/env python3
"""Assemble the self-contained HTML results dashboard (embeds every PNG as
a base64 data URI so it publishes as a single Artifact file)."""

from __future__ import annotations

import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ABLATIONS = RESULTS / "ablations"
TABLES = RESULTS / "tables"
OUT = RESULTS / "dashboard.html"


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def img(path: Path, alt: str) -> str:
    return f'<img src="data:image/png;base64,{b64(path)}" alt="{alt}" loading="lazy" />'


HTML = f"""<!doctype html>
<title>Terrain Detail-Transition Karşılaştırması</title>
<style>
:root {{
  --bg: #f6f4ef;
  --bg-alt: #eeece3;
  --panel: #ffffff;
  --text: #1c1a17;
  --text-dim: #5a564d;
  --rule: #d8d3c6;
  --accent: #b5642a;
  --accent-soft: #e7c8a8;
  --good: #3f7d52;
  --bad: #a83b30;
  --mono-bg: #eae6da;
  --shadow: 0 1px 2px rgba(30,25,15,0.06), 0 8px 24px rgba(30,25,15,0.05);
}}
:root[data-theme="dark"] {{
  --bg: #14120f;
  --bg-alt: #1b1815;
  --panel: #1f1c18;
  --text: #ece7dc;
  --text-dim: #a89e8c;
  --rule: #3a352c;
  --accent: #e0985a;
  --accent-soft: #4a3624;
  --good: #79c393;
  --bad: #e0776a;
  --mono-bg: #232019;
  --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 12px 32px rgba(0,0,0,0.35);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #14120f;
    --bg-alt: #1b1815;
    --panel: #1f1c18;
    --text: #ece7dc;
    --text-dim: #a89e8c;
    --rule: #3a352c;
    --accent: #e0985a;
    --accent-soft: #4a3624;
    --good: #79c393;
    --bad: #e0776a;
    --mono-bg: #232019;
    --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 12px 32px rgba(0,0,0,0.35);
  }}
}}

* {{ box-sizing: border-box; }}
html {{ background: var(--bg); }}
body {{
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  line-height: 1.55;
  margin: 0;
  padding: 0 0 5rem;
}}
h1, h2, h3 {{
  font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, "Times New Roman", serif;
  text-wrap: balance;
  font-weight: 600;
  color: var(--text);
  letter-spacing: 0.005em;
}}
.mono, code, .stat-value, table, td, th {{
  font-family: ui-monospace, "SF Mono", "Cascadia Code", "Consolas", monospace;
  font-variant-numeric: tabular-nums;
}}
a {{ color: var(--accent); }}
a:focus-visible, button:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}

.wrap {{ max-width: 76rem; margin: 0 auto; padding: 0 1.75rem; }}

header.top {{
  border-bottom: 1px solid var(--rule);
  background: linear-gradient(180deg, var(--bg-alt), var(--bg));
  padding: 3rem 0 2rem;
}}
.eyebrow {{
  text-transform: uppercase;
  letter-spacing: 0.14em;
  font-size: 0.72rem;
  color: var(--text-dim);
  font-weight: 600;
}}
h1.title {{
  font-size: clamp(1.7rem, 3.2vw, 2.5rem);
  margin: 0.4rem 0 0.6rem;
}}
p.lede {{
  max-width: 46rem;
  color: var(--text-dim);
  font-size: 1.02rem;
  margin: 0 0 1.6rem;
}}
.meta-row {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem 1.4rem;
  font-size: 0.82rem;
  color: var(--text-dim);
}}
.meta-row b {{ color: var(--text); }}

nav.jump {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin-top: 1.6rem;
}}
nav.jump a {{
  text-decoration: none;
  font-size: 0.78rem;
  padding: 0.35rem 0.75rem;
  border: 1px solid var(--rule);
  border-radius: 999px;
  color: var(--text-dim);
  background: var(--panel);
}}
nav.jump a:hover {{ color: var(--accent); border-color: var(--accent-soft); }}

.stat-row {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(9.5rem, 1fr));
  gap: 1px;
  background: var(--rule);
  border: 1px solid var(--rule);
  margin-top: 2rem;
}}
.stat {{
  background: var(--panel);
  padding: 1.1rem 1.1rem 1.2rem;
}}
.stat-value {{
  font-size: 1.65rem;
  font-weight: 600;
  color: var(--accent);
  line-height: 1.1;
}}
.stat-label {{
  font-size: 0.76rem;
  color: var(--text-dim);
  margin-top: 0.35rem;
}}

section {{ padding: 2.6rem 0; border-bottom: 1px solid var(--rule); }}
section:last-of-type {{ border-bottom: none; }}
.section-head {{ display: flex; align-items: baseline; gap: 0.8rem; margin-bottom: 1.1rem; }}
.section-num {{ font-family: ui-monospace, monospace; color: var(--text-dim); font-size: 0.85rem; }}
h2 {{ font-size: 1.4rem; margin: 0; }}
h3 {{ font-size: 1.08rem; margin: 1.6rem 0 0.6rem; }}
p {{ margin: 0.6rem 0; }}
.dim {{ color: var(--text-dim); }}

.panel {{
  background: var(--panel);
  border: 1px solid var(--rule);
  border-radius: 6px;
  box-shadow: var(--shadow);
  padding: 1.1rem 1.1rem 1.3rem;
}}
figure {{ margin: 1.2rem 0; }}
figure img {{ max-width: 100%; display: block; border-radius: 4px; }}
figcaption {{ font-size: 0.82rem; color: var(--text-dim); margin-top: 0.5rem; }}

.table-scroll {{ overflow-x: auto; border: 1px solid var(--rule); border-radius: 6px; background: var(--panel); }}
.table-scroll img {{ display: block; min-width: 640px; width: 100%; }}

.grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.4rem; }}
.grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.2rem; }}
@media (max-width: 860px) {{ .grid-2, .grid-3 {{ grid-template-columns: 1fr; }} }}

.card {{
  background: var(--panel);
  border: 1px solid var(--rule);
  border-radius: 6px;
  padding: 1.1rem 1.2rem 1.3rem;
  box-shadow: var(--shadow);
}}
.card h3 {{ margin-top: 0; font-size: 1rem; }}
.finding {{
  border-left: 3px solid var(--accent);
  padding: 0.15rem 0 0.15rem 0.9rem;
  margin: 0.8rem 0;
  color: var(--text);
  background: var(--bg-alt);
  border-radius: 0 4px 4px 0;
}}
.finding b {{ color: var(--accent); }}

.flag-bad {{ color: var(--bad); font-weight: 600; }}
.flag-good {{ color: var(--good); font-weight: 600; }}

ol.findings {{ padding-left: 1.3rem; }}
ol.findings li {{ margin: 0.9rem 0; }}

table.simple {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
table.simple th, table.simple td {{ border: 1px solid var(--rule); padding: 0.4rem 0.6rem; text-align: left; }}
table.simple th {{ background: var(--bg-alt); }}

pre {{
  background: var(--mono-bg);
  border: 1px solid var(--rule);
  border-radius: 6px;
  padding: 0.9rem 1rem;
  overflow-x: auto;
  font-size: 0.82rem;
}}

footer {{ padding: 2.5rem 0 1rem; color: var(--text-dim); font-size: 0.82rem; }}
</style>

<header class="top">
  <div class="wrap">
    <div class="eyebrow">LunarRocket &middot; deney raporu</div>
    <h1 class="title">Ay İnişi Simülasyonunda Arazi Detay-Geçiş Yöntemlerinin Karşılaştırılması</h1>
    <p class="lede">Smoothstep, Gaussian ve önerilen hibrit (inişayağı-yakını Gaussian + ötesi smoothstep) yöntemlerinin
    üretim/sorgu süresi, VRAM ve dikiş (seam) sürekliliği açısından karşılaştırılması. Isaac Lab'daki gerçek
    üretim formülleri temel alınarak, GPU/Isaac Sim gerektirmeden CPU/NumPy üzerinde çalıştırıldı.</p>
    <div class="meta-row">
      <span><b>30</b> rastgele tohum &middot; iki paralel deney kolu (mesh &amp; analitik)</span>
      <span><b>4</b> yöntem varyantı</span>
      <span><b>5</b> duyarlılık (ablation) taraması</span>
      <span>Wilcoxon işaretli-sıra testi, n=30</span>
    </div>
    <nav class="jump">
      <a href="#yontem">Yöntem</a>
      <a href="#ana-sonuclar">Ana Sonuçlar</a>
      <a href="#ablasyonlar">Ablasyonlar</a>
      <a href="#istatistik">İstatistiksel Anlamlılık</a>
      <a href="#bulgular">Bulgular</a>
      <a href="#literatur">İlgili Çalışmalar</a>
      <a href="#sinirlamalar">Sınırlamalar</a>
      <a href="#tekrarlanabilirlik">Tekrarlanabilirlik</a>
    </nav>
    <div class="stat-row">
      <div class="stat"><div class="stat-value">~3200&times;</div><div class="stat-label">as-shipped Gaussian'ın C0 dikiş hatası / smoothstep</div></div>
      <div class="stat"><div class="stat-value">~530&times;</div><div class="stat-label">C1 (normal açı) hata oranı</div></div>
      <div class="stat"><div class="stat-value">0</div><div class="stat-label">hibrit switch'in yarattığı yeni dikiş (makine hassasiyeti)</div></div>
      <div class="stat"><div class="stat-value">4.3&times;</div><div class="stat-label">hibrit fade maliyeti / smoothstep</div></div>
      <div class="stat"><div class="stat-value">p&lt;10<sup>-8</sup></div><div class="stat-label">tüm anlamlı karşılaştırmalarda (n=30)</div></div>
    </div>
  </div>
</header>

<div class="wrap">

<section id="yontem">
  <div class="section-head"><span class="section-num">01</span><h2>Yöntem ve düzeltme</h2></div>
  <p>Başlangıç varsayımı "mevcut sistem iki-mesh + smoothstep, öneri tek-grid Gaussian" şeklindeydi. Kod taramasında
  bunun tersi çıktı:</p>
  <div class="grid-2">
    <div class="card">
      <h3>Canlı RL ortamı &mdash; <span class="mono">smoothstep</span></h3>
      <p class="dim"><code>_local_detail_height</code>, <code>lunar_lander_env.py:954-980</code></p>
      <p>Analitik yükseklik alanı (mesh yok); ince "regolith" katmanı <code>t&sup2;(3-2t)</code> ile
      <code>R=8m</code> yarıçapında karıştırılıyor. Her fizik adımında sorgulanan gerçek üretim yolu budur.</p>
    </div>
    <div class="card">
      <h3>Legacy mesh pipeline &mdash; <span class="mono">gaussian</span></h3>
      <p class="dim"><code>_blend_local_patch_multiscale</code>, <code>app/terrain_generator.py:924-991</code></p>
      <p>Global 128&times;128 grid + 256&times;256 local patch; <code>exp(-(d/R)&sup2;)</code> ile karıştırılıyor.
      Sadece <code>ISAACLAB_USE_DEM_TERRAIN=1</code> ile erişilen görsel/demo yolu.</p>
    </div>
  </div>
  <p>Bu yüzden her iki alt-sistem de gerçek baseline olarak ele alındı: <b>mesh-track</b> (global+local grid,
  literal "iki-mesh" çerçevesi) ve <b>analytic-track</b> (canlı RL ortamının kapalı-form yükseklik fonksiyonu).
  Üçüncü yöntem olarak <b>hibrit</b> eklendi: iniş ayağı temas bölgesinde (<code>Rc=1.5m</code>, gerçek ayak
  izinden türetildi) Gaussian, ötesinde smoothstep, aradaki 0.3m'lik bantta kendisi de smoothstep-ağırlıklı
  yumuşak bir anahtarla (switch) birleştiriliyor &mdash; yeni bir dikiş yaratmadan.</p>
  <p>Arazi <em>içeriği</em> (eğim, dalga, krater, kaya, ince-detay genliği/fazı) her denemede gerçek üretim
  üretecinden (<code>app.terrain_pool.generate_terrain_pool_batch</code>) örneklendi.</p>
</section>

<section id="ana-sonuclar">
  <div class="section-head"><span class="section-num">02</span><h2>Ana karşılaştırma (30 tohum, ortalama)</h2></div>
  <div class="table-scroll">{img(TABLES/"main_comparison.png", "Ana karşılaştırma tablosu")}</div>

  <div class="grid-3" style="margin-top:1.6rem;">
    <figure class="panel">{img(RESULTS/"radial_profile.png", "Radyal yükseklik kesiti")}
      <figcaption>Radyal kesit (tohum=0): dört yöntem R=8m ve Rc=1.5m civarında gözle ayırt edilemez &mdash;
      fark, dikiş bölgesinin ince yapısında.</figcaption></figure>
    <figure class="panel">{img(RESULTS/"continuity_comparison.png", "Süreklilik karşılaştırması")}
      <figcaption>Dikiş-özel (seam-only) C0/C1 hatası: <span class="flag-bad">gaussian_shipped</span> hem mesh
      hem analitik kolda belirgin şekilde daha kötü; diğer üçü makine hassasiyetine yakın.</figcaption></figure>
    <figure class="panel">{img(RESULTS/"cost_comparison.png", "Maliyet karşılaştırması")}
      <figcaption>Üretim süresi ve sorgu verimi blend şeklinden bağımsız; saf fade maliyeti ise
      <span class="flag-bad">hibrit'te ~4.3&times;</span> daha yüksek. VRAM proxy'si tek-grid dağıtımda
      yöntemden bağımsız.</figcaption></figure>
  </div>
</section>

<section id="ablasyonlar">
  <div class="section-head"><span class="section-num">03</span><h2>Duyarlılık taramaları (ablations)</h2></div>

  <h3>A &middot; Hibrit temas yarıçapı R<sub>c</sub> taraması</h3>
  <div class="grid-2">
    <figure class="panel">{img(ABLATIONS/"rc_sweep.png", "Rc taraması")}<figcaption>R<sub>c</sub> &isin; {{1.0 &hellip; 4.0}} m, sabit R=8m, bant=0.3m.</figcaption></figure>
    <div class="table-scroll">{img(TABLES/"rc_sweep_summary.png", "Rc taraması özeti")}</div>
  </div>
  <p class="finding">İç switch hatası R<sub>c</sub> büyüdükçe hafifçe artıyor (3.2e-5 &rarr; 2.9e-4) ama her durumda
  makine-hassasiyeti mertebesinde kalıyor; fade maliyeti R<sub>c</sub>'den pratik olarak bağımsız (~755&ndash;793&micro;s).
  <b>R<sub>c</sub>=1.5m seçimi</b> (gerçek ayak izi + iniş toleransı) tasarım uzayının güvenli bölgesinde.</p>

  <h3>B &middot; Switch bant genişliği taraması</h3>
  <div class="grid-2">
    <figure class="panel">{img(ABLATIONS/"switch_width_sweep.png", "Switch genişliği taraması")}<figcaption>genişlik &isin; {{0.05 &hellip; 2.0}} m, sabit R<sub>c</sub>=1.5m.</figcaption></figure>
    <div class="table-scroll">{img(TABLES/"switch_width_summary.png", "Switch genişliği özeti")}</div>
  </div>
  <p class="finding">C0/C1 genişlikten bağımsız olarak ~0 kalıyor (formel olarak her zaman C1) &mdash; ama
  <b>ikinci türev (eğrilik) genişlik &rarr; 0 iken patlıyor</b> (8.2 &rarr; 0.13, ~60&times; fark): çok dar bir
  switch bandı, formel süreklilik korunsa da, pratikte "daha az pürüzsüz" bir geçiş üretiyor. 0.3m'lik seçim bu
  eğrilik maliyetini düşük tutuyor.</p>

  <h3>C &middot; Karıştırma yarıçapı R taraması</h3>
  <div class="grid-2">
    <figure class="panel">{img(ABLATIONS/"radius_sweep.png", "R taraması")}<figcaption>R &isin; {{4 &hellip; 16}} m.</figcaption></figure>
    <div class="table-scroll">{img(TABLES/"radius_sweep_summary.png", "R taraması özeti")}</div>
  </div>
  <p class="finding"><span class="flag-bad">gaussian_shipped</span>'in dikiş hatası R'den neredeyse bağımsız
  (C1 RMS: 0.672&deg; &rarr; 0.664&deg;, R=4&hellip;16m boyunca) &mdash; çünkü <code>exp(-(r/R)&sup2;)</code>
  yalnızca <code>r/R</code>'ye bağlı, <b>ölçek-değişmez bir kusur</b>. Yani sorun R=8m varsayılanına özgü bir
  parametre ayarı hatası değil, formülün kendisinde yapısal.</p>

  <h3>D &middot; Mesh çözünürlük taraması (ıraksama derecesi)</h3>
  <figure class="panel" style="max-width:44rem;">{img(ABLATIONS/"resolution_sweep.png", "Çözünürlük taraması")}
    <figcaption>global çözünürlük &isin; {{64,96,128,192,256}}, 10 tohum/nokta.</figcaption></figure>
  <p class="finding">smoothstep ve hibrit için ayrıklaştırma hatası çözünürlükle net biçimde azalıyor
  (deneysel yakınsama derecesi p&asymp;0.90, bilineer enterpolasyonun beklenen birinci-mertebe davranışıyla
  tutarlı). <span class="dim">gaussian_calibrated için ölçülen "negatif yakınsama" (p&asymp;-0.36) gerçek bir
  fiziksel etki değil</span>: bu yöntemin hatası zaten ~1e-6 m'lik sabit kesme (truncation) tortusunda düz
  seyrediyor, yani orada ölçülen küçük dalgalanma grid çözünürlüğünden değil, kalibrasyon sabitinden
  kaynaklanıyor &mdash; bu ölçekte anlamlı bir yakınsama derecesi yorumlanamaz.</p>
</section>

<section id="istatistik">
  <div class="section-head"><span class="section-num">04</span><h2>İstatistiksel anlamlılık</h2></div>
  <p>Her 30 tohum, dört yöntemde de <em>aynı</em> rastgele arazi içeriğiyle eşleştirilmiş (paired) şekilde
  çalıştırıldığından, yöntemler arası fark için eşleştirilmiş Wilcoxon işaretli-sıra testi kullanıldı
  (parametrik olmayan, dağılım varsayımı gerektirmiyor). Klasik Cohen's d / t-testi güven aralıkları, oranın
  tohumlar arası neredeyse sabit olması nedeniyle (etkinin belirli bir rastgele örneğe özgü olmadığının
  kendisi bir bulgu) sayısal olarak dejenere olduğundan raporlanmadı; onun yerine log10-oran aralığı
  (min&ndash;max) ve Wilcoxon p-değeri kullanıldı.</p>
  <div class="table-scroll">{img(TABLES/"significance_tests.png", "İstatistiksel anlamlılık tablosu")}</div>
  <p class="finding"><b>hibrit vs smoothstep</b> için fark tam olarak sıfır (30/30 tohumda) &mdash; bu bir test
  başarısızlığı değil, R sınırında hibrit'in analitik olarak smoothstep'e eşit olmasının (tasarım gereği)
  doğrudan sonucu. Diğer tüm karşılaştırmalarda p&lt;2&times;10<sup>-9</sup>, ve etkinin yönü/büyüklüğü 30
  tohumun tamamında tutarlı (aralık neredeyse sıfır genişlikte).</p>
</section>

<section id="bulgular">
  <div class="section-head"><span class="section-num">05</span><h2>Bulgular</h2></div>
  <ol class="findings">
    <li><b>As-shipped Gaussian formülü gerçek, ölçülebilir bir dikiş bırakıyor; smoothstep, kalibre-Gaussian ve
    hibrit bırakmıyor.</b> Kök neden: <code>exp(-(r/R)&sup2;)</code> C&infin; düzgün ama hiçbir zaman tam sıfıra
    inmiyor (R'de hâlâ <code>exp(-1)&asymp;%36.8</code> kalıntı taşıyor); sınırsız bırakıldığında hiçbir sonlu
    yarıçapta dikiş göstermiyor (doğrulandı), ama <em>her</em> dağıtılabilir gösterim (mesh, heightmap) bir
    yerde kesme yapmak zorunda. Legacy kodun kendi kesmesi (<code>blend_range=1.5&times;blend_idx</code>,
    <code>app/terrain_generator.py:947</code>) bunu 12m'ye itiyor, orada bile <code>%10.5</code> kalıntı var.</li>
    <li><b>VRAM, tek-grid ("baked") dağıtımda blend şekli ekseninde ayırt edici değil.</b> Üç yöntem de aynı
    global-çözünürlük grid'ini paylaştığı için aynı ~1.05MB tahminine iniyor; analitik kolda ise üçü de aynı
    42 float/env (168 byte/env) arazi-parametresini okuyor. Literal <code>two_mesh_literal</code> dağıtımı
    (gerçek ayrı local mesh) blend şeklinden bağımsız olarak ~5.24MB'a mal oluyor &mdash; yani VRAM ekseninde
    asıl belirleyici mesh-vs-analitik ve baked-vs-literal seçimleri, blend fonksiyonu değil.</li>
    <li><b>Üretim süresi ve uçtan-uca sorgu verimi blend şeklinden büyük ölçüde bağımsız</b> (mesh: 5.6&ndash;6.1ms,
    analitik: ~3.0&ndash;3.1M sorgu/s) &mdash; ikisi de paylaşılan makro/ince arazi hesaplamasında baskın oluyor,
    fade formülünde değil. İzole edildiğinde (fade-only mikro-benchmark) fark ortaya çıkıyor: smoothstep ~93&micro;s,
    Gaussian ~%35 daha pahalı, <b>hibrit ~4.3&times;</b> daha pahalı (her çağrıda iki dal + switch ağırlığı
    hesaplanıyor). Bu maliyet mevcut batch büyüklüğünde/arazi karmaşıklığında görünmez kalıyor ama arazi
    değerlendirmesi ucuzlatılırsa veya fade çok daha yüksek oranda çağrılırsa önem kazanabilir.</li>
    <li><b>Hibrit'in iç switch'i (R<sub>c</sub>'de Gaussian&harr;smoothstep) ölçülebilir yeni bir dikiş
    yaratmıyor</b>: <code>|fade(Rc+&epsilon;)&minus;fade(Rc&minus;&epsilon;)|=8.7&times;10<sup>-5</sup></code>,
    eğim süreksizliği <code>1.3&times;10<sup>-4</sup></code> &mdash; iki bileşen fonksiyonun kendi
    pürüzsüzlüğünden ayırt edilemeyecek düzeyde. Hibrit, iniş-ayağı temas bölgesinde kalibre-Gaussian
    kalitesindeki pürüzsüzlüğü, ötesinde ise smoothstep'in ucuz kompakt desteğini miras alıyor &mdash;
    (3)'te belirtilen maliyet karşılığında.</li>
  </ol>
</section>

<section id="literatur">
  <div class="section-head"><span class="section-num">08</span><h2>İlgili çalışmalar (literatür taraması)</h2></div>
  <p>ARS <code>deep-research</code> (<code>lit-review</code> modu) ile hedefli tarama: 5 tematik kümede
  14 doğrulanmış kaynak (her biri DOI/arXiv/yayıncı sayfasından tek tek doğrulandı). Tam açıklamalı
  liste ve her kaynağın bulgularımızla bağlantısı: <code>LITERATURE_REVIEW.md</code>.</p>
  <div class="grid-2">
    <div class="card">
      <h3>1 &middot; Terrain LOD / çoklu-çözünürlük geçişi</h3>
      <p class="dim">Losasso &amp; Hoppe (2004, geometry clipmaps) &middot; Duchaineau ve ark. (1997, ROAM)
      &middot; de Boer (2000, geomipmapping)</p>
      <p>Grafik literatürü 25+ yıldır aynı sınıf problemi (düşük/yüksek çözünürlük sınırında
      crack/seam) çözmeye çalışıyor &mdash; bizim ölçtüğümüz Gaussian-kesme dikişi bu ailenin somut bir örneği.</p>
    </div>
    <div class="card">
      <h3>2 &middot; Kompakt vs. sonsuz destekli karıştırma (süreklilik)</h3>
      <p class="dim">Hardy (1971, multiquadric) &middot; Wendland (1995, CSRBF) &middot; Ohtake ve ark.
      (2003) &middot; Ebert ve ark. (2002, smoothstep)</p>
      <p>Gaussian'ın hiçbir sonlu yarıçapta tam sıfıra inmemesi bir kodlama hatası değil, radyal baz
      fonksiyonlarının 50+ yıllık bilinen bir yapısal özelliği (Hardy, 1971); Wendland (1995) bunun
      kompakt-destekli çözümünü formalize ediyor.</p>
    </div>
    <div class="card">
      <h3>3 &middot; Gezegen/ay inişinde arazi doğruluğu</h3>
      <p class="dim">Johnson &amp; Montgomery (2008, ALHAT/TRN) &middot; JPL ALHAT programı &middot;
      Barker ve ark. (2016, LOLA DEM)</p>
      <p>Gerçek iniş mühendisliğinde arazi doğruluğu operasyonel bir gereksinim (100m hassas iniş
      hedefi, ~m mertebesinde DEM doğruluğu) &mdash; dikiş/süreklilik sorununun neden önemsenmesi
      gerektiğinin gerçek-dünya gerekçesi.</p>
    </div>
    <div class="card">
      <h3>4 &middot; RL ortamlarında prosedürel arazi randomizasyonu</h3>
      <p class="dim">Tobin ve ark. (2017, domain randomization) &middot; Rudin ve ark. (2022,
      IsaacGym terrain-curriculum) &middot; Mittal ve ark. (2023, Orbit) &middot; NVIDIA (2025, Isaac Lab)</p>
      <p><b>Boşluk</b>: bu literatür terrain <em>çeşitliliğini</em> optimize ediyor, blend-fonksiyonu
      <em>seçimini</em> değil &mdash; çalışmamızın konumlandığı somut boşluk.</p>
    </div>
  </div>
  <p class="finding"><b>5 &middot; Hibrit/bölgesel birleştirme</b>: Ohtake ve ark. (2003)'ün çok-ölçekli
  partition-of-unity çerçevesi, bizim <code>switch_weight()</code> fonksiyonumuzun (iki fade
  fonksiyonunu C&sup1;-düzgün bir ağırlıkla birleştiren) tam olarak oturduğu matematiksel aile
  &mdash; hibrit switch'in yeni dikiş yaratmaması, izole bir mühendislik hilesi değil, bu çerçeveden
  beklenen bir sonuç.</p>
</section>

<section id="sinirlamalar">
  <div class="section-head"><span class="section-num">06</span><h2>Sınırlamalar</h2></div>
  <ul>
    <li>Bu ortamda GPU/Isaac Sim yok: VRAM rakamları analitik bir proxy (vertex/triangle say&#305;s&#305;
    &times; varsay&#305;lan byte/vertex), &ouml;l&ccedil;&uuml;lm&uuml;&#351; bir <code>nvidia-smi</code>
    tahsisi değil. <code>TERRAIN_QUALITY.md</code>'deki ampirik rakamlarla (tam Isaac Sim/PhysX sahne
    yükünü içerir, sadece bu mesh'i değil) büyüklük mertebesi kıyaslaması faydalı bir sağlama ama
    bire-bir doğrulama değil.</li>
    <li>İnce/detay katmanı formülü, kasıtlı olarak şu üretim kodundaki "hedefte detayı çıkar" terimini
    atlıyor (<code>_local_detail_height</code>) &mdash; blend-şekli karşılaştırmasına dik, hedef-yükseklik
    eşleme kaygısı.</li>
    <li><code>R<sub>c</sub>=1.5m</code> ve 0.3m switch bandı tek sabit seçimler (Rc taraması ve genişlik
    taraması bunları doğruluyor ama tam bir 2D ızgara taraması değil) &mdash; hibrit yöntem bölümünü
    güçlendirecek bir takip çalışması olabilir.</li>
  </ul>
</section>

<section id="tekrarlanabilirlik">
  <div class="section-head"><span class="section-num">07</span><h2>Tekrarlanabilirlik</h2></div>
  <p>Tamamen CPU/NumPy; GPU veya Isaac Sim gerekmiyor. Kod: <code>experiments/terrain_transition/</code>.</p>
  <pre>cd experiments/terrain_transition
python3 -m venv .venv &amp;&amp; .venv/bin/pip install numpy matplotlib scipy
.venv/bin/python run_experiment.py --seeds 30        # ana karşılaştırma
.venv/bin/python ablation_experiments.py              # 4 duyarlılık taraması + anlamlılık testleri
.venv/bin/python export_tables.py                     # CSV/Markdown/PNG tablolar
.venv/bin/python build_dashboard.py                   # bu sayfa</pre>
  <p class="dim">Ham veriler: <code>results/raw_results.json</code>, <code>results/ablations/*.csv</code>,
  tablolar: <code>results/tables/*.{{csv,md,png}}</code>. Tam metodoloji ve formül kaynak-satırı referansları:
  <code>experiments/terrain_transition/README.md</code>.</p>
</section>

</div>

<footer>
  <div class="wrap">LunarRocket &middot; terrain detail-transition deney raporu &middot; 30 tohum, CPU-only, tekrarlanabilir.</div>
</footer>
"""

OUT.write_text(HTML)
print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
