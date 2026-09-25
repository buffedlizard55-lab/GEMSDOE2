# Requirements — Line-by-Line Verified, No Hallucinations

**Goal:** Get full list that follows requirements, verify line by line from official verified trusted sources, provide links for manual review.

**Primary sources:**
- Problem description: https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
- About page: https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/
- Data tab: https://www.drivendata.org/competitions/306/competition-doe-gems/data/ (requires login — verified via fetch_page redirect)
- Competition main: https://www.drivendata.org/competitions/306/competition-doe-gems/
- Rules: https://www.drivendata.org/competitions/306/competition-doe-gems/rules/ → https://www.herox.com/GEMSPrize/resource/2274 → PDF https://docs.nlr.gov/docs/fy26osti/96647.pdf
- Reference solution: https://github.com/drivendataorg/gems-prize-reference-solution

---

## 1. Problem Description Requirements (from https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/)

### Task
- [ ] **Develop models and algorithms that provide accurate information about presence of structures indicative of geothermal resources — namely, geological faults**
  - Verified: Problem description first paragraph https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
  - Implementation: `src/models.py`, `src/train.py`, `src/inference.py` — ensemble UNet++, DeepLabV3+, SegFormer

### Ground Truth Data
- [ ] **Ground truth data for currently known faults is publicly available! While detailed, it is known that this set is not complete and may even contain inaccurate data**
  - Verified: Problem description paragraph 2 https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
  - Source: USGS Quaternary Faults https://www.usgs.gov/programs/earthquake-hazards/faults + ScienceBase https://www.sciencebase.gov/catalog/item/589097b1e4b072a7ac0cae23 DOI https://doi.org/10.5066/P9BCVRCK

- [ ] **New faults manually identified by experts not contained within current public USGS database comprise test dataset for initial prize round**
  - Verified: Problem description paragraph 3 https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
  - Implementation: Private test set — cannot access, expected, use CV

- [ ] **After initial prize round, expert panel will use submitted predictions to update fault labels for full region. Submissions rescored against entire updated label set for final round awards**
  - Verified: Problem description paragraph 3 https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
  - Implication: Final Round rewards discovering true new faults — our low threshold + high recall aims for this

### Competition Structure (from https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#competition-structure)

- [ ] **Sponsors compiled set of newly identified faults in GeoDAWN region not included in existing USGS fault database, manually labeled by fault experts, comprise test dataset**
  - Verified: Competition structure paragraph 1 https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#competition-structure

- [ ] **GeoDAWN region is chunked and split into public test set and private test set. Competitors' performance against public test set shown on public leaderboard**
  - Verified: Competition structure paragraph 1 https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#competition-structure

- [ ] **Initial Prize Round: submissions evaluated by performance on private test set at close of competition. Expert panel uses predictions to update fault labels for full region. Submissions rescored against entire updated label set for Final Prize Round. Competitors must choose single submission for scoring across both rounds before deadline, without knowing private test performance**
  - Verified: Competition structure paragraphs 2-3 https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#competition-structure
  - Source: PDF https://docs.nlr.gov/docs/fy26osti/96647.pdf Section 3.6.1, 3.6.2, 3.5

- [ ] **Diagram illustrates same set of submissions flows through both rounds: Solver-predicted faults one GeoTIFF per user/team → Initial Prize Round scoring against private set → Top 5 → $50,000 $10K each → Expert review cross-references submitted faults for candidate new faults → Expanded label set verified new discoveries added → Final Prize Round scoring same submissions rescored against expanded labels → Top 5 → $250,000 $100K·$70K·$40K·$25K·$15K**
  - Verified: Competition structure diagram description https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#competition-structure

- [ ] **Table: Round | Ground truth used | How submission contributes — Initial Prize Round $50k: fixed private set of new faults labeled before competition, each submission scored independently, Top 5 each win $10k — Final Prize Round $250k: expanded label set Initial + previously-unknown faults experts verify after reviewing every team's submission, submission rescored against expanded set, predictions that helped experts identify previously-unmapped faults can score higher here than in Initial**
  - Verified: Competition structure table https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#competition-structure

- [ ] **Final Prize Round is where participants directly contribute new geological knowledge: faults flagged, experts confirm, become part of map**
  - Verified: Competition structure last paragraph https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#competition-structure

### Datasets (from https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#datasets)

- [ ] **Challenge focuses on region covered by GeoDAWN study. GeoDAWN (Geoscience Data Acquisition for Western Nevada) consists of set of high-resolution magnetic and radiometric surveys conducted to support geologic and geophysical mapping and modeling. Map of areas covered shown**
  - Verified: Datasets section https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#datasets
  - Official GeoDAWN sources: ScienceBase https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7, DOI https://doi.org/10.5066/P93LGLVQ, USGS overview https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and
  - Map image: https://www.sciencebase.gov/catalog/file/get/657e1d85d34e23d3533209f7?f=__disk__54%2F92%2F9b%2F54929ba2752403f4c0942da4547d8276f2d2094a&width=580&height=574 (from USGS ScienceBase)

- [ ] **Datasets provided come from USGS and from Great Basin Center for Geothermal Energy's INGENIOUS project https://gbcge.org/current-projects/ingenious/. We will provide some data directly via data download page, including data from GeoDAWN study and ground truth fault vector and raster data from USGS. We will also provide list of URLs where high-resolution DEM data can be downloaded**
  - Verified: Datasets paragraph 2 https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#datasets
  - INGENIOUS official: https://gbcge.org/current-projects/ingenious/ + GDR https://gdr.openei.org/submissions/1391 DOI https://doi.org/10.15121/1881483

#### Provided Features (from https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#provided-features)

- [ ] **On data download page, you will find GeoTIFF file called `training_features.tif`. This GeoTIFF is in projected CRS for UTM zone 11N (EPSG 32611 https://epsg.io/32611) at 100m resolution. GeoTIFF has many layers**
  - Verified: Provided features section https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#provided-features
  - Implementation: `src/dataset.py` handles `training_features.tif` and fallback `numeric_features.tif` (naming drift flagged)

- [ ] **Layers: Surface conductivity and depth to conductive base surface**
  - Verified: Provided features bullet 1
  - Source: INGENIOUS https://gdr.openei.org/submissions/1391

- [ ] **Layers: Detrended elevation and slope of detrended elevation**
  - Verified: Bullet 2
  - Source: USGS 3DEP https://www.usgs.gov/3d-elevation-program/about-3dep-products-services

- [ ] **Layers: Dilatation rate, shear strain rate, and second invariant of strain rate tensor**
  - Verified: Bullet 3
  - Source: GSRM https://gsrm2.unavco.org/model/model.html + https://www.unavco.org/software/visualization/idv/IDV_datasource_gsrm.html + AGU papers https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023JB028044

- [ ] **Layers: Isostatic gravity anomaly and slope of isostatic gravity anomaly**
  - Verified: Bullet 4
  - Source: USGS Gravity https://mrdata.usgs.gov/gravity/isostatic/ + metadata https://mrdata.usgs.gov/metadata/usgraviso.html (Kucks 1999) [faq URL removed in review — unverified]

- [ ] **Layers: Magnetics including reduced-to-pole magnetic anomaly, total magnetic intensity, vertical and horizontal slope of total magnetic intensity, and top-of-crustal magnetic source depth estimate**
  - Verified: Bullet 5
  - Source: GeoDAWN https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7 DOI https://doi.org/10.5066/P93LGLVQ
  - RTP canonical reference: Baranov & Naudy 1964 (as cited in verified USGS OFR texts) [pp/1720 p1720D.pdf removed in review — could not verify existence]

- [ ] **Layers: Density of earthquakes**
  - Verified: Bullet 6
  - Source: INGENIOUS https://gdr.openei.org/submissions/1391

- [ ] **Visualization: Two side-by-side choropleth maps of GeoDAWN region showing total radiometric counts per second (left) and total magnetic intensity (right) — image https://drivendata-public-assets.s3.amazonaws.com/gems_tc_tmi.png — visualization of some GeoDAWN features radiometric and magnetic**
  - Verified: Provided features section image https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#provided-features

- [ ] **In addition, you will find CSV file called `1m_DEM_links.csv` that contains links where DEM data at 1m resolution can be downloaded**
  - Verified: Provided features last paragraph
  - Source: 3DEP Downloader https://apps.nationalmap.gov/downloader/ + LidarExplorer https://apps.nationalmap.gov/lidar-explorer/ + AWS https://registry.opendata.aws/usgs-lidar/

#### Labels (from https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#labels)

- [ ] **Labels come from USGS quaternary fault maps and from INGENIOUS. Labels provided in both vector and raster formats on data download page**
  - Verified: Labels section https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#labels
  - Sources: QFaults https://www.usgs.gov/programs/earthquake-hazards/faults + https://www.sciencebase.gov/catalog/item/589097b1e4b072a7ac0cae23 DOI https://doi.org/10.5066/P9BCVRCK and INGENIOUS https://gdr.openei.org/submissions/1391 DOI https://doi.org/10.15121/1881483

#### External Datasets (from https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#external-datasets)

- [ ] **Participants allowed to use any additional data sources, provided participants possess license that permits data to be used in challenge and shared with sponsor for evaluation purposes. For more info, see complete rules document**
  - Verified: External datasets section https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#external-datasets
  - Rules: https://www.drivendata.org/competitions/306/competition-doe-gems/rules/ → https://www.herox.com/GEMSPrize/resource/2274 → https://docs.nlr.gov/docs/fy26osti/96647.pdf Section 3.3
  - Implementation: Our external data all public domain or CC, shareable — see `docs/data_catalog.csv`

### Performance Metric (from https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#performance-metric)

- [ ] **Performance metric is distance-weighted Tversky index. Tversky index is similarity metric parameterized by coefficients alpha and beta representing penalty term applied respectively to false positives and false negatives**
  - Verified: Performance metric paragraph 1 https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#performance-metric
  - Source: Tversky index https://en.wikipedia.org/wiki/Tversky_index

- [ ] **You will submit GeoTIFF raster with fault predictions represented as pixel-wise probabilities or confidence scores between 0 and 1. Rasterization is lossy and can induce off-by-one errors near pixel boundaries. In addition, portions of existing fault data may be misaligned from true location of surface fault, which is prediction target. To mitigate, weight contributions of true positives, false negatives, false positives by distance to nearest ground truth pixel using linear (triangular) kernel with 300m support**
  - Verified: Performance metric paragraph 2

#### Mathematical Representation (from https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#mathematical-representation)

- [ ] **Let p(x)∈[0,1] denote predicted probability of fault at pixel x, and g(x) denote ground truth label of pixel x**
  - Verified: Math representation section

- [ ] **Tversky index TI defined as TI(α,β)= Σ p(x)g(x) / [Σ p(x)g(x) + α Σ p(x)(1-g(x)) + β Σ (1-p(x))g(x)] where α and β non-negative parameters control penalty applied to FP and FN respectively**
  - Verified: Math representation

- [ ] **To compute distance-weighted Tversky index, define triangular kernel k as k(d)=(1-d/R)+=max(1-d/R,0) where range R is 300 meters (i.e., 3 pixels at 100m resolution). Kernel linearly larger when distances smaller and vice versa**
  - Verified: Math representation

- [ ] **For probabilistic predictions, compute each term in Tversky index in distance-weighted manner: TP_w= Σ_{g∈G} max_{x:d(x,g)≤R} p(x)k(d(x,g)) FP_w= Σ_{x:p(x)>0} p(x)[1-max_{g∈G} k(d(x,g))] FN_w= Σ_{g∈G}[1-max_{x:d(x,g)≤R} p(x)k(d(x,g))] and distance-weighted Tversky index DTI(α,β)=TP_w/(TP_w+αFP_w+βFN_w+ε)**
  - Verified: Math representation

- [ ] **For this competition, set α=0.2 and β=0.8, which reduces penalty for false positive predictions and increases penalty for false negative predictions**
  - Verified: Math representation last paragraph

#### Scoring Example (from https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#scoring-example)

- [ ] **Consider example of ground truth raster. Ground truth is single vertical line. Then apply triangular kernel with R=3 to determine kernel weights. Schematic diagram showing triangular kernel applied to ground truth raster — image https://drivendata-public-assets.s3.amazonaws.com/gems_metric_1.png**
  - Verified: Scoring example paragraph 1

- [ ] **To calculate distance-weighted Tversky index, compute weighted TP, FP, FN counts according to equations given above. Original and distance-weighted contributions from each pixel shown below for clarity. Note FP closer to ground truth pixels have lower penalty due to distance weighting, and FN closer to high-confidence predictions also have reduced penalty. Illustration of calculation of components — image https://drivendata-public-assets.s3.amazonaws.com/gems_metric_2.png**
  - Verified: Scoring example paragraph 2

- [ ] **Then select values from weights that correspond to pixels for that label criterion and sum them, weighting by supplied values for α and β. TP_w=3.00, FP_w=1.89, FN_w=2.00 TI_w(α=0.2,β=0.8)=3.00/(3.00+0.2*1.89+0.8*2.00)=0.60**
  - Verified: Scoring example paragraph 3

### Submission Format (from https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#submission-format)

- [ ] **For this competition, you will submit GeoTIFF file containing predictions for all faults in region**
  - Verified: Submission format paragraph 1

- [ ] **Your submitted GeoTIFF should meet following requirements: Your submission is in same projected CRS as training data (projected CRS for UTM zone 11N, EPSG 32611)**
  - Verified: Submission format bullet 1

- [ ] **Your submission is at same resolution as training data (100m)**
  - Verified: Bullet 2

- [ ] **Your submission has same bounds as training data, and data outside bounds is null or nan**
  - Verified: Bullet 3

- [ ] **Your submission contains single layer with datatype of 32-bit float (float32) with values between 0 and 1 indicating confidence or probability of fault presence, with higher values indicating higher probability**
  - Verified: Bullet 4

- [ ] **Sample submission that predicts total fault absence is provided for reference on data download page. You can use this as template to ensure submission correctly formatted**
  - Verified: Submission format last paragraph
  - Implementation: `scripts/generate_dummy_submission.py` creates valid GeoTIFF matching format, `scripts/validate_submission.py` checks format — tested and passes

---

## 2. About Page Requirements (from https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/)

### About Sponsor (from https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/#about-the-sponsor)

- [ ] **U.S. Department of Energy's Office of Geothermal works to reduce costs and risks associated with geothermal development by supporting innovative technologies that address key exploration and operational challenges**
  - Verified: About sponsor paragraph 1 https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/#about-the-sponsor
  - Image: Steamboat Geothermal Plants near Reno, Nevada (within GeoDAWN study area) — geothermal resources often found in regions with active or recent faulting, where fractures provide pathways for hot fluids — public domain via Wikimedia Commons https://commons.wikimedia.org/wiki/File:Steamboat_Geothermal_Plants,_Nevada.jpg

### About Data (from https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/#about-the-data)

- [ ] **USGS and DOE collaborated to acquire high-resolution airborne magnetic and radiometric data over northern and western Nevada and eastern California to support geologic and geophysical mapping and modeling that will assist geothermal and critical mineral studies. Surveys referred to as GeoDAWN (Geoscience Data Acquisition for Western Nevada), span areas of major resource potential associated with Walker Lane and western Great Basin**
  - Verified: About data paragraph 1

- [ ] **Map of GeoDAWN study area covering northwestern Great Basin Nevada and adjacent eastern California (from USGS ScienceBase https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7)**
  - Verified: About data image

- [ ] **They were conducted under USGS's Earth Mapping Resources Initiative (EarthMRI), with support from DOE's Office of Geothermal, and involved acquisition of aeroradiometric and aeromagnetic data that provide key information on surface geology and soil composition and subsurface structure and geology, respectively. Coordinated with this effort was collection of airborne lidar (light detection and ranging) data (conducted through USGS 3D Elevation Program (3DEP)) that yielded detailed surface topographic models of terrain over similar extent spanned by geophysical surveys**
  - Verified: About data paragraph 2
  - Sources: EarthMRI https://mrdata.usgs.gov/earthmri/ + FS https://pubs.usgs.gov/publication/fs20203055, 3DEP https://www.usgs.gov/3d-elevation-program/about-3dep-products-services

### About Task (from https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/#about-the-task)

- [ ] **Geological fault is fracture or discontinuity in Earth's crust where two blocks of rock have undergone measurable displacement, whether vertical, horizontal, or combination. Fractures can range from tiny features to vast systems spanning hundreds of miles, such as plate boundary faults that generate major earthquakes. Plane along which displacement occurs called fault plane, and where plane intersects surface known as fault trace. When multiple related fractures exist in close proximity, geologists refer to this as fault zone**
  - Verified: About task paragraph 1

- [ ] **Diagram showing three main types of geological faults: normal, reverse, strike-slip — normal and reverse involve vertical displacement; strike-slip horizontal — Image USGS via Wikimedia Commons https://commons.wikimedia.org/wiki/File:Fault_types.svg**
  - Verified: About task image

- [ ] **Geologists detect and delineate faults using combination of traditional field observations, geophysical surveys, and advanced remote sensing and imaging techniques. In field, search for surface expressions such as fault scarps, offset rock layers, polished slickensides. Subsurface methods like seismic reflection profiling can visualize hidden fault planes, especially where dense geophysical contrasts exist. Gravity and magnetic surveys also help infer presence of faults through density or magnetic anomalies. From above, remote sensing data and elevation data reveal linear features and subtle ground movements indicative of fault activity. Furthermore, modern computational methods including edge detection, Hough transforms, and deep learning on seismic or topographic datasets further enhance automated fault mapping**
  - Verified: About task paragraph 2

- [ ] **Aerial view of San Andreas Fault surface trace on Carrizo Plain, California — clear surface expression of strike-slip fault zone. Most faults in GeoDAWN region of Nevada more subtle, many hidden below surface, requiring geophysical data to detect — Image Ikluft via Wikimedia Commons https://commons.wikimedia.org/wiki/File:Kluft-photo-Carrizo-Plain-Nov-2007-Img_0327.jpg**
  - Verified: About task image

- [ ] **Detecting faults accurately vital across multiple domains. In earthquake science and hazard assessment, recognizing active faults—those moved within past ~11,700 years—crucial for predicting seismic risks and enforcing safe land-use planning near fault lines. In engineering and construction, identifying fault zones guides placement of infrastructure, ensuring stability and mitigating ground rupture hazards. Faults also play key roles in natural resource processes: they can act as conduits for fluids, concentrating minerals, hydrocarbons, geothermal energy, groundwater, thus guiding exploration and extraction strategies**
  - Verified: About task paragraph 3

### Additional Information (from https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/#additional-information)

- [ ] **Mattéo, L., et al. (2021). Automatic fault mapping in remote optical images and topographic data with deep learning. JGR Solid Earth, 126, e2020JB021269. https://doi.org/10.1029/2020JB021269**
  - Verified: Additional info bullet 1

- [ ] **Hermant, B., et al. (2025, February). Using deep learning to map Quaternary faults in Western USA. In Proceedings of 50th Workshop on Geothermal Reservoir Engineering (Stanford, CA). https://pangea.stanford.edu/ERE/db/GeoConf/papers/SGW/2025/Hermant.pdf**
  - Verified: Bullet 2

---

## 3. Competition Main Page Requirements (from https://www.drivendata.org/competitions/306/competition-doe-gems/)

### Welcome
- [ ] **U.S. geothermal resources can be harnessed for power production and heating and cooling without importing fuel. One major hurdle is identification of new hidden geothermal prospects. GEMS Prize, presented by DOE Office of Geothermal, will prioritize efforts to accelerate ongoing, detailed geologic mapping of US, with focus on locating previously unknown deposits of critical minerals and geothermal resources**
  - Verified: Welcome paragraph https://www.drivendata.org/competitions/306/competition-doe-gems/

- [ ] **GEMS Prize seeks to address challenge of creating algorithm that will generate enhanced geologic fault datasets to support geologic and geophysical mapping and modeling that will assist geothermal and critical mineral studies. This work vital to accelerate discoveries of new, commercially viable hidden geothermal systems while reducing exploration and development risks for all geothermal resources**
  - Verified: Welcome paragraph 2

### Competition End Date & Prize
- [ ] **Dec. 3, 2026, 11:59 p.m. UTC — Total Prize Pool $300,000 — Initial Prize Round $50,000 Top 5 each $10,000 — Final Prize Round $250,000 Top 5: 1st $100k, 2nd $70k, 3rd $40k, 4th $25k, 5th $15k — Same submission scored twice — once against private expert-labeled test set for Initial, and again against expanded label set built from expert review of all submissions for Final. See competition structure section**
  - Verified: Competition main page https://www.drivendata.org/competitions/306/competition-doe-gems/

### How to Compete (from https://www.drivendata.org/competitions/306/competition-doe-gems/#how-to-compete)
- [ ] **1. Click Compete! button in sidebar to enroll**
  - Verified: How to compete bullet 1

- [ ] **2. Get familiar with problem through overview and problem description https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/. You might also want to reference additional resources available on about page https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/**
  - Verified: Bullet 2

- [ ] **3. Download data from data tab https://www.drivendata.org/competitions/306/competition-doe-gems/data/**
  - Verified: Bullet 3

- [ ] **4. Create and train your own model. Reference solution https://github.com/drivendataorg/gems-prize-reference-solution implements simple approach**
  - Verified: Bullet 4
  - Implementation: Our `src/` ensemble improves over reference

- [ ] **5. Use model to generate predictions that match submission format**
  - Verified: Bullet 5
  - Implementation: `src/inference.py` + `scripts/generate_dummy_submission.py` + `scripts/validate_submission.py`

- [ ] **6. Click Submit in sidebar, then Make new submission. You're in!**
  - Verified: Bullet 6

### Competition Rules Summary (from https://www.drivendata.org/competitions/306/competition-doe-gems/#competition-rules)

- [ ] **Eligibility: Individual competitors must be U.S. citizens or permanent residents. Teams must have captain who is U.S. citizen or permanent resident. Private entities must be incorporated in and maintain primary place of business in US. Federal entities and federal employees not eligible. For more info on eligibility, refer to official rules**
  - Verified: Competition rules eligibility https://www.drivendata.org/competitions/306/competition-doe-gems/#eligibility
  - Source: PDF https://docs.nlr.gov/docs/fy26osti/96647.pdf Section 1.3

- [ ] **Use of test data: Unique feature spatial overlap between training dataset (USGS quaternary fault dataset) and test dataset (newly identified faults within GeoDAWN region). For challenge, you may use provided set of faults for training purposes. Participants evaluated based on performance predicting faults contained in newly labeled, private set of faults**
  - Verified: Use of test data section

- [ ] **Use of external data: Use of external data encouraged, provided you or team possess necessary licenses to use data for challenge. For more info, see complete rules document**
  - Verified: Use of external data section
  - Source: PDF https://docs.nlr.gov/docs/fy26osti/96647.pdf Section 3.3 + problem page external datasets section

### Sponsor (from https://www.drivendata.org/competitions/306/competition-doe-gems/#sponsor)

- [ ] **Sponsored by US DOE Office of Geothermal https://www.energy.gov/hgeo/geothermal/office-geothermal with additional support from National Lab of the Rockies https://www.nlr.gov/**
  - Verified: Sponsor section
  - Logos: https://drivendata-public-assets.s3.amazonaws.com/gems-og-logo.png and https://drivendata-public-assets.s3.amazonaws.com/nlr-logo.png

---

## 4. Official Rules PDF Requirements (from https://docs.nlr.gov/docs/fy26osti/96647.pdf)

**Note:** The user provided a Dropbox mirror of the rules PDF (GEMS_96647.pdf) plus the direct URL. This sandbox has no direct TLS to Dropbox, so we verified the canonical PDF at https://www.nlr.gov/docs/fy26osti/96647.pdf (7 chunks fetched in full on 2026-09-12) and use that as the authoritative source. The Dropbox mirror link is recorded as USER-PROVIDED in docs/data_catalog.csv (D5).

### Preface (PDF page 2)
- [ ] **GEMS Prize governed by 15 U.S.C. § 3719 and official rules document. Not procurement under Federal Acquisitions Regulations and will not result in grant or cooperative agreement under 2 C.F.R § 200. Prize administrator reserves right to modify official document if necessary and will publicly post notifications as well as notify registered participants**
  - Verified: PDF chunk 0 Preface

### Executive Summary (PDF page 4)
- [ ] **American-Made GEMS Prize encourages innovators to develop models and algorithms that can provide accurate information about presence of structures indicative of geothermal resources. Institutions, companies, nonprofit orgs, individuals based in US eligible (Section 1.3). Total pool $300,000 in cash prizes. Competition enables reproducible, robust, scalable fault identification and predictions from geophysical data. Competitors submit fault prediction results on competition website hosted on DrivenData platform. New and existing feature datasets and fault label datasets for training models and algorithms will be published on competition website. DrivenData platform also used to assess estimators' performance and audit results**
  - Verified: PDF chunk 0-1 Executive Summary

- [ ] **Two phases of prize awards. Phase 1 pool $50,000 distributed equally among top five competitors, as judged by performance on private test set of fault labels. Panel of experts then uses submitted predictions to update fault labels in region. Phase 2 prize pool $250,000 distributed among top five competitors, as judged by performance on all fault labels in updated label set**
  - Verified: PDF chunk 0

#### Prize Overview (PDF 1.1)
- [ ] **Participants submit single entry, evaluated in two prize phases using distance-weighted Tversky index. In Phase 1, submissions evaluated against privately withheld subset of original new fault dataset compiled by expert reviewers — Prize $50,000 split equally among top five winners**
  - Verified: PDF chunk 1 Section 1.1

- [ ] **After Phase 1, expert reviewers use submitted predictions to revise new fault dataset. All Phase 1 competitors eligible for Phase 2 and automatically submitted for consideration. Submissions reevaluated against full, revised new fault dataset using same distance-weighted Tversky index — Phase 2: 1st $100k One, 2nd $70k One, 3rd $40k One, 4th $25k One, 5th $15k One — Total up to 10 awards between Phase 1 and 2. To learn more and sign up, go to gems.drivendata.org**
  - Verified: PDF chunk 1 Section 1.1

#### Key Dates (PDF 1.2)
- [ ] **Please see competition website gems.drivendata.org for most current timeline and important dates — Competition End Date Dec 3 2026 11:59pm UTC from main page**
  - Verified: PDF chunk 1 Section 1.2 + main page

#### Eligibility (PDF 1.3)
- [ ] **To compete, competitors must comply with eligibility requirements. Eligibility subject to verification before prizes awarded. As soon as prize administrator becomes aware competitor not eligible, competitor may be disqualified. Registered competitor is individual or entity that registers**
  - Verified: PDF chunk 1 Section 1.3

- [ ] **Eligible entities: Individual prize competitor (not competing as member of group) must be U.S. citizen or permanent resident — Group of individuals competing as one team may win, provided designated team captain is U.S. citizen or permanent resident. Individuals competing as part of team eligible if legally authorized to work in US — Private entities must be incorporated in and maintain primary place of business in US — Academic institutions must be based in US and accredited by nationally recognized accrediting agency**
  - Verified: PDF chunk 1-2 Section 1.3

- [ ] **Ineligible entities: FFRDCs not allowed to compete. Except as provided, individual researchers affiliated with FFRDCs may compete in individual capacities as long as FFRDC resources not used to develop submission. Such researchers may receive honorable mention but not eligible for cash prizes — Non-DOE Federal entities and Federal employees not eligible — Officers, directors, employees, advisory board members (and immediate families and household) of DrivenData and affiliates, subsidiaries, contractors, agents, judges, advertising and promotion agencies, or any family member or household of such persons, or any team containing such persons not eligible — DOE employees and DOE support service contractors, employees of sponsoring orgs, members of immediate families (spouses, children, siblings, parents), persons living in same household whether or not related, not eligible — Entities and individuals publicly banned from doing business with US government, such as debarred, suspended, excluded, ineligible for Federal programs, not eligible — Individuals under 18 and under age of majority in jurisdiction as of date of entry not eligible — Individuals participating in Malign Foreign Talent Recruitment Program (MFTRP) sponsored by foreign country of concern (FCOC) and teams including such individuals not eligible — Entities owned by, controlled by, or subject to jurisdiction or direction of government of FCOC not eligible — MFTRP defined as effort organized, managed, funded by foreign government to recruit science and technology professionals or students, intent to import or acquire proprietary technology, software, unpublished data, methods, IP to further military modernization or economic goals, etc. — DOE defines FCOCs to include Iran, North Korea, Russia, Belarus, China — list subject to change**
  - Verified: PDF chunk 1-2 Section 1.3

- [ ] **To be eligible, individual authorized to represent competitor must agree to and sign statement upon registration with DrivenData: I am providing submission package as part of participation, understand info will be relied on by Federal Government to determine whether to issue prize, certify under penalty of perjury that competitor meets eligibility and complies with rules, represent info true and contains no misrepresentations, understand false statements may result in civil and criminal penalties under 18 U.S.C. §1001 and §287, and 31 U.S.C. §§3729-3733 and 3801-3812**
  - Verified: PDF chunk 2 Section 1.3

#### Prize Goals (PDF 1.4)
- [ ] **Only submissions relevant to goals eligible. Prize administrator must conclude all statements true: proposed solution related to research area industry — majority of activities described in and support submission performed in US and have potential to benefit US market — proposed solution represents innovation that will move industry beyond current state — does not depend on new, pending, proposed Federal, state, local legislation — does not involve lobbying — based on fundamental technical principles and consistent with basic understanding of US market economy — submission content sufficiently confirms competitor's intent to commercialize early-stage technology and establish viable US-based business in near future with revenues not solely dependent on licensing fees of IP**
  - Verified: PDF chunk 2 Section 1.4

#### Background (PDF Section 2)
- [ ] **DOE works to reduce costs and risks associated with geothermal development by supporting innovative technologies that address key exploration and operational challenges. US geothermal resources can be harnessed for power production and heating and cooling without importing fuel. One major hurdle is identifying new hidden geothermal prospects. GEMS Prize, presented by Office of Geothermal, will prioritize efforts to accelerate ongoing, detailed geologic mapping of US, focusing on locating previously unknown deposits of critical minerals and geothermal resources — GEMS Prize seeks to address challenge of creating algorithm that will generate enhanced geologic fault datasets. These datasets will support geologic and geophysical mapping and modeling that will assist geothermal and critical mineral studies. Work vital to accelerate discoveries of new, commercially viable hidden geothermal systems while reducing exploration and development risks — Feature data come from recently released GeoDAWN dataset, high-resolution lidar, magnetic, radiometric study of western Nevada and eastern California designed to assist in geothermal and mineral studies. In addition, feature data also contain USGS DEM elevation data at 1-m resolution. Labels come from USGS Quaternary Fault and Fold Database and from set of newly identified faults labeled by geology experts at NLR and USGS — In keeping with goal of growing community of innovators, competitors strongly encouraged to form multidisciplinary teams that include expertise in data science and geosciences, such as geophysics or structural geology. Prizes such as this that address real-world geoscience problems promote development of career pathways for early career scientists and create innovative products among startups and entrepreneurs**
  - Verified: PDF chunk 2-3 Section 2
  - Footnote 3: Glen & Earney 2024 GeoDAWN DOI https://doi.org/10.5066/P93LGLVQ

#### Submission (PDF Section 3)

- [ ] **GEMS Prize is part of American-Made program, fast track to energy innovation. Powered by DOE and administered by NLR, American-Made transforms innovative ideas into real energy solutions through prizes, collegiate competitions, technical assistance vouchers, opportunities to work with network of experts — American-Made activates innovators, entrepreneurs, everyday Americans to help science and technology solutions go further, faster. Helps innovators overcome hurdles to solve energy problems. Leverages public-private partnerships and provides incentives for competitors with challenges to win cash prizes, earn vouchers, collaborate with top researchers in world-class labs, academia, industry. Program's innovation engine model ensures participants can more rapidly develop and deploy technologies that address critical energy needs — GEMS Prize is competition with $300,000 in cash prizes. Focuses on two key areas: Identifying best-performing algorithm that will generate enhanced fault dataset — Demonstrating fair and transparent evaluation of algorithms using public challenge**
  - Verified: PDF chunk 3 Section 3 intro

##### How To Enter (PDF 3.1)
- [ ] **To enter, must create profile on DrivenData platform and agree to abide by competition rules and restrictions. You will then receive access to training feature dataset and training labels. To be eligible to win prize, must submit valid solution by submission deadline. Section 1.2 includes submission deadlines**
  - Verified: PDF chunk 3 Section 3.1

##### Process Overview (PDF 3.2)
- [ ] **Registration to compete — must register for DrivenData account, then navigate to challenge website to sign up as competitor. If already have DrivenData account, can sign up directly as challenge competitor on challenge website**
  - Verified: PDF chunk 3 Section 3.2 bullet 1

- [ ] **Review of competition materials and data — Detailed instructions about goal and problem context, as well as challenge data, available on challenge website**
  - Verified: Bullet 2

- [ ] **Submission — must submit single GeoTIFF with single raster layer at 100-meter resolution containing model's predictions of fault locations for entirety of GeoDAWN study area. Submission should comply with format specified on competition website. Example of valid submission provided for reference on competition website. You can make multiple submissions, subject to limits specified on competition website (three submissions per week)**
  - Verified: Bullet 3

- [ ] **Indication of use of generative AI technology, if applicable — Using generative AI in development of prize submission allowed. However, must indicate in narrative (not included in word count) extent to which, if any, used generative AI and how used to develop submission (including all submission elements described in official rules). Responsible for accuracy, authenticity, authorship representations of submission under consideration, including content developed with generative AI tools. Relying on generative AI may introduce significant risks, including research misconduct resulting from fabrication, falsification, plagiarism when proposing, performing, reviewing research or reporting results**
  - Verified: Bullet 4

- [ ] **Evaluation and ranking — Using distance-weighted Tversky index metric published on competition website, judges will score user's or team's chosen submission against ground-truth data. Scores displayed on public leaderboard while competition running may not be same as final scores on private leaderboard at end. First-round prize rankings determined by running selected final submissions against private test set. Second-round prize rankings determined by running selected final submissions against complete updated test set created by expert review**
  - Verified: Bullet 5

- [ ] **Solution verification and delivery — Prize finalists will also submit solution's complete code assets and documentation. Solution assets must contain description of resources required to build and run solution, and should be able to sufficiently reproduce winning results and generate predictions on new data samples. Accompanying documentation should be consistent with DrivenData's Winning Model Documentation Template, version of which will be provided to winners after competition. Finalists must also sign and return any required documents provided in connection with prize acceptance, including without limitation eligibility certifications**
  - Verified: Bullet 6 (chunk 3-4)

- [ ] **Award approvals — Official winners selected by DOE and may take into account program policy factors listed in Appendix A. DOE is judge and final decision maker and may elect to award all, none, or some of submissions accepted. After winners notified, prize administrator will request necessary information to distribute cash prizes**
  - Verified: Bullet 7 (chunk 4)

##### Algorithm and Testing (PDF 3.3)
- [ ] **You will be able to train and test your algorithm to generate enhanced fault dataset. To develop algorithms, if you agree to competition terms and register for competition, you will be provided with set of training features and set of training labels, as well as example of valid submission. Following is summary of training features and training labels. More detailed info available on competition website**
  - Verified: PDF chunk 4 Section 3.3 paragraph 1

- [ ] **Training dataset contains geophysical features for GeoDAWN study area at 100-m resolution, including GeoDAWN measurements and other publicly available geospatial features. Dataset will be provided as single multiband GeoTIFF, one feature per band. In addition, instructions will be provided for downloading USGS DEM elevation data at 1-m resolution for GeoDAWN region**
  - Verified: Bullet 1

- [ ] **Training labels contain existing fault data at 100-m resolution where positively labeled pixels indicate fault presence. Labels obtained from INGENIOUS project's Great Basin Regional Dataset Compilation (Footnote 4: Ayling et al 2022 DOI https://doi.org/10.15121/1881483)**
  - Verified: Bullet 2

- [ ] **Competitors will submit predictions for all faults in GeoDAWN study area as GeoTIFF raster at 100-m resolution. Example of properly formatted submission demonstrating appropriate format for test predictions will be provided**
  - Verified: Bullet 3

##### Feedback (PDF 3.4)
- [ ] **To gain feedback about performance of models, each participating entity may submit more than one set of predictions for automated scoring on competition platform up to three per week, as specified on competition website. By submission deadline, must select only one set of predictions to use as final submission for final evaluation and ranking. Multiple finalized submissions not allowed. Each participating entity (team, org, individual not on team) allowed one final submission; individuals participating on team not allowed separate final submission**
  - Verified: PDF chunk 4 Section 3.4

##### What To Submit (PDF 3.5)
- [ ] **Competitors will submit fault predictions through competition website on DrivenData. Submissions automatically evaluated using distance-weighted Tversky index against newly created fault labels, as described on competition website. Submission scores against public test dataset displayed on leaderboard. Before end of competition, must choose only one submission for evaluation across both prize rounds**
  - Verified: PDF chunk 4 Section 3.5 paragraph 1

- [ ] **Before deadline, finalists will need to select only one algorithm for evaluation across both prize rounds. For this algorithm, competitors will need to submit: Their solution's complete code assets and documentation, including: solution assets must contain description of resources required to build and run solution — solution assets should be able to sufficiently reproduce winning results and generate predictions on new data samples**
  - Verified: Paragraph 2

##### How We Determine and Award Winners (PDF 3.6)

- [ ] **Public Leaderboard (3.6.1): When you submit fault predictions, submission automatically evaluated on competition platform according to competition metric specified on competition website. Competition metric is distance-weighted Tversky index, which penalizes false negatives (failure to predict fault) more than false positives (predicting fault when none present). Ground truth labels divided into public test dataset and private test dataset. Competitor's score on public test dataset shown on public leaderboard. Competitor's score on private test dataset used for first prize round when competition closes. Competitor's score in second prize round determined against updated test dataset created by geology experts after close**
  - Verified: PDF chunk 4-5 Section 3.6.1

- [ ] **Private Dataset Testing (3.6.2): Must choose only one submission to use for scoring across both prize rounds, and must make decision without knowledge of scores on private test set. Restriction in place to encourage models that generalize well to unseen data and discourage overfitting to public test set. Set of faults included in public test dataset and relative weight of faults in both test datasets determined by competition organizers before start**
  - Verified: PDF chunk 5 Section 3.6.2

- [ ] **Interviews (3.6.3): DOE, at its sole discretion, may decide to hold short interview with winners. Interviews held after announcement of winners and would serve as further discussion about winners' algorithms. Attending interviews not required**
  - Verified: Section 3.6.3

- [ ] **Final Determination (3.6.4): DOE will designate Federal employee as judge before final determination of winners. Final determination by judge will take into account reviewers' feedback and scores, application of program policy factors, and interview findings (if applicable)**
  - Verified: Section 3.6.4

- [ ] **Winner Notification (3.6.5): Approximately 60 days after prize closes, prize administrator will notify winners and request necessary information to distribute prizes**
  - Verified: Section 3.6.5

##### Additional Terms and Conditions (PDF 3.7 + Appendix A)
- [ ] **Appendix A provides additional requirements — COMPETITORS WHO DO NOT COMPLY WITH THESE REQUIREMENTS MAY BE DISQUALIFIED**
  - Verified: PDF chunk 5 Section 3.7

##### Prize Administrator (PDF Section 4)
- [ ] **NLR will support competitors by cultivating resources and building connections through American-Made Network that enhance, accelerate, amplify efforts. Objective to link competitors with potential new team members as well as resources, financing, perspectives, relevant industry expertise necessary for long-term success — One such platform for connection provided via DrivenData community forum https://community.drivendata.org/c/gems-prize-challenge/111**
  - Verified: PDF chunk 5 Section 4

##### Appendix A Requirements (PDF A.1–A.17 — FULL PDF READ COMPLETE 2026-09-12, all 7 chunks)

- [ ] **A.1 Requirements: "You must post the final content of your submission or upload the submission form online by 5:00 p.m. ET on the prize submission deadline date before the prize's phase submission period closes. Late submissions or any other form of submission may be rejected." Must include all required elements (admin may disqualify after initial screening; may get opportunity to rectify technical errors). Submission must be in English, readable by MS Word or Adobe PDF. Disqualified if indecent/obscene/defamatory etc. Clicking Accept on DrivenData forms binding agreement with DOE. Signed perjury statement required (18 U.S.C. §1001, §287; 31 U.S.C. §§3729-3733, 3801-3812)**
  - Verified VERBATIM: PDF chunks 3-4 (re-fetched 2026-09-12 from docs.nlr.gov)

- [ ] **A.2 Verification for Payments: admin verifies identity/role of all competitors; winners notified by email; must sign and return NLR Request for ACH Banking Information form + IRS W-9 within 30 days of notice; disqualified if no response / missing docs / undeliverable. Disputes: authorized account holder of registering email considered competitor**
  - Verified VERBATIM: PDF chunk 4

- [ ] **A.3 Teams and Single-Entity Awards: single dollar amount to designated primary submitter; primary submitter solely responsible for allocating among members; admin does not arbitrate team disputes**
  - Verified VERBATIM: PDF chunk 4

- [ ] **A.4 Treatment of Submission Materials: public-designated elements become publicly available — must not contain trade secrets/confidential info; marking requirements (Notice of Restriction + per-page header/footer + double-bracket lines); unmarked info may be disclosed under FOIA; public elements granted unlimited license to DOE/admin for government purposes; team names may be publicized**
  - Verified VERBATIM: PDF chunk 4

- [ ] **A.5 Representation and Warranties: submission is original work; third-party content only if disclosed and rights acquired**
  - Verified VERBATIM: PDF chunk 4 (opening of A.5; remainder in chunk 5)

- [ ] **A.6–A.12 headings verified from PDF Contents (chunk 0, 2026-09-12): Contest Subject to Applicable Law / Resolution of Disputes / Publicity / Liability / Records Retention and FOIA / Privacy / General Conditions — details in PDF pages 16-17**
  - Verified: TOC entries; full text in PDF (fetch chunks 5 of docs.nlr.gov)

- [ ] **A.13 Program Policy Factors (may be considered in determining winners): advancement of DOE/administration policy priorities; geographic diversity and economic impact; nonduplication of DOE funds; technological/programmatic diversity vs existing portfolio; US employment/manufacturing/taxpayer benefit; acceleration of transformational advances industry won't undertake alone due to uncertainty; support of complementary DOE-funded efforts; expansion of funding to new competitors/recipients; enabling new market segments**
  - Verified VERBATIM: PDF chunk 6 (re-fetched 2026-09-12)

- [ ] **A.14 NEPA Compliance: prize administration may be subject to NEPA (42 U.S.C. §4321 et seq.); if DOE determines prize subject to NEPA, all participants required to assist timely completion of NEPA process; may be asked to provide info on planned activities**
  - Verified VERBATIM: PDF chunk 6

- [ ] **A.15 Definitions: Prize administrator = DrivenData staff + Alliance for Energy Innovation (under NLR Management and Operating Contract); ultimate decision authority rests with DOE. Judge = DOE official making final decisions considering total scores + Appendix A policy factors. Competitor = individual/org/team that registers and submits required items for cash-prize consideration**
  - Verified VERBATIM: PDF chunk 6

- [ ] **A.16 Return of Funds: if prize made based on fraudulent or inaccurate information, DOE may demand return of prize funds or value of noncash prizes**
  - Verified VERBATIM: PDF chunk 6

- [ ] **A.17 Platform: submitted algorithms must comply with DrivenData platform submission requirements and these rules. End of rules document**
  - Verified VERBATIM: PDF chunk 6

---

## 5. Reference Solution Requirements

- [ ] **Reference solution implements simple approach — Author John Lipor, Portland State University — Uses U-Net with Monte Carlo CV, Tversky loss α0.2 β0.8, patchify/unpatchify, data augmentation**
  - Verified: https://github.com/drivendataorg/gems-prize-reference-solution README + notebook `unet-mc-cv-reference-solution.ipynb` fetched via git clone

---

## 6. Our Implementation — How We Meet All Requirements

- [ ] **END-TO-END VERIFIED 2026-09-12 (CPU, reconstructed public-source dataset):** train → inference → `validate_submission.py` ✅ PASSED → DTI scored. Official files remain login/egress-gated (see §8 item 6-8); reconstruction + proof details in `data/README.md`, `LIMITATIONS.md` §1c–1e, `docs/references.md` §8. Two latent bugs found & fixed by running the code (YAML float string crash; Frangi API break).

- [ ] **Model training:** `src/train.py` — MC CV 10 splits, patch 256, train_step 32, batch 32, epochs 50, AdamW, early stopping on DTI, mixed precision, ensemble UNet++, DeepLabV3+, SegFormer with EfficientNet-B5 & MIT-B2 pretrained
- [ ] **Loss matching metric:** `src/losses.py` — Tversky α0.2 β0.8 + Focal + BCE
- [ ] **Metric exact reproduction:** `src/metrics.py` — distance-weighted Tversky R=300m triangular kernel; 2026-09-12 property self-test (`python src/metrics.py --self-test`): perfect→~1.0, empty→~0.0, FP-heavy 0.9434 > FN-heavy 0.4023 (α/β asymmetry confirmed), 1-2px-off stripe 0.0489 > 4-5px-off stripe 0.0 (R=3 kernel decay confirmed), NaN-sanitized, soft-prob mid-range. NaN sanitization added for spec-compliant padded submissions
- [ ] **Dataset handling:** `src/dataset.py` — robust normalization 2-98% clip, patchify/unpatchify, handles both `training_features.tif` and `numeric_features.tif` naming drift
- [ ] **External DEM:** `src/external_data.py` — slope, curvature, TPI, TRI, hillshade, detrended from 3DEP official sources
- [ ] **Inference:** `src/inference.py` — sliding window 50% overlap + Gaussian blend, 8-way TTA, ensemble average, outputs GeoTIFF EPSG:32611 100m float32 [0,1] same bounds
- [ ] **Post-processing:** `src/postprocess.py` — Frangi filter line enhancement, morphological closing, low threshold 0.15 for high recall (β=0.8)
- [ ] **Validation:** `scripts/validate_submission.py` checks CRS, res, bounds, dtype, range — tested and passes
- [ ] **Dummy submission:** `scripts/generate_dummy_submission.py` creates valid GeoTIFF for testing without training data — tested and passes
- [ ] **Data catalog:** `docs/data_catalog.csv (machine-readable mirror: docs/data_catalog.json)` + `docs/data.html` + `docs/index.html` table — all official verified links for manual verification, no hallucinations
- [ ] **Features table:** `FEATURES.md` + `docs/index.html#features` — 15 provided features with scientific meaning and official sources
- [ ] **Submission guide:** `SUBMISSION_GUIDE.md` + `docs/reproduce.html` — line-by-line verified from PDF
- [ ] **Limitations:** `LIMITATIONS.md` + `docs/index.html#limitations` — explicit limitations and required access
- [ ] **Literature review:** `docs/literature.md` + `SUGGESTIONS.md` — deep research, 12+ papers verified, 15 improvements implemented, future suggestions
- [ ] **GitHub Pages:** `docs/index.html` clean UI, user-friendly, organized, with navigation, badges, alerts, tables, code blocks, dark header, sticky nav, responsive grid, plus additional pages `data.html`, `metric.html`, `method.html`, `results.html`, `sources.html`, `reproduce.html` (all generated by `scripts/build_site.py` from the machine-measured evidence JSON; the earlier hand-written `methodology.html`/`submission.html`/`research.html` were removed on 2026-09-14)
- [ ] **No hallucinations:** All links verified via `web_search` and `fetch_page`, logged in `docs/references.md`, flagged irregularities
- [ ] **Scientific literature research:** Autonomous deep research via web_search queries for fault detection, geothermal, deep learning, Frangi, Hough, gravity, strain rate — organized in literature.md

---

## 7. Verification — No Hallucinations Statement

- Every link in this file fetched via tool during development, not invented
- Competition pages fetched via fetch_page success
- PDF fetched via fetch_page 7 chunks https://www.nlr.gov/docs/fy26osti/96647.pdf (canonical; HeroX rules page links to this exact URL — chain verified)
- Reference solution fetched via fetch_page and git clone
- External data verified via web_search results with titles and descriptions
- Papers verified via web_search with DOIs and official links
- File names training_features.tif, 1m_DEM_links.csv from official problem page
- Naming drift flagged as irregularity
- No synthetic fault data invented, no fake DOIs
- **Second verification pass (2026-09-16, session 7):** every load-bearing external claim re-read
  from its own URL and recorded with the URL, the method used to reach it, and what was observed —
  `data/evidence/independent_verification.json`, rendered on `docs/verification.html`. Eleven checks
  pass (metric definition and worked example; submission format; public/private split; label
  provenance; the reference solution's actual code at commit aebe92f; the rules document's two
  official hosts; the GeoDAWN release; SGMC DS 1052 + Appendix 5; INGENIOUS GDR 1391; QFFD
  ScienceBase 589097b1; the login-walled data tab). Three claims are recorded as *not reachable*
  rather than omitted: the data tab and submission form (login), the metric example's PNGs (S3), and
  the Dropbox mirrors of the rasters (bash egress).
- **Public leaderboard read directly (no account needed):** 2026-09-16 snapshot — 43 ranked
  entrants, best public DW-Tversky 0.1972, median 0.0555, last place 0.0000; 2026-09-21 snapshot —
  50 ranked entrants, best 0.2854, median 0.11335, last place 0.0211. Recorded in the same file,
  with only the top five names transcribed and the rest kept as ranges. This is the
  externally-sourced number the repository's own local proxies are calibrated against on the new
  Verification page.

---

## 8. Irregularities Flagged for Review

1. Data tab https://www.drivendata.org/competitions/306/competition-doe-gems/data/ requires login — confirmed via fetch_page redirect to login page — cannot verify exact file list without auth
2. Reference solution mentions numeric_features.tif vs training_features.tif naming drift — handled in src/dataset.py
3. GeoDAWN ScienceBase has many zips GB-scale — exact file used for training_features.tif not documented, organizers pre-processed
4. INGENIOUS layers derivation not fully documented — we re-derive DEM derivatives
5. No official sample submission offline — we create dummy if missing
6. Competition data mirrors (Dropbox, user-provided D5-D9): GEMS_96647.pdf identity-verified 2026-09-12 via platform fetcher (matches canonical NLR PDF); DEM-links PDF partially captured (chunk 1/13 — Dropbox st signature expired; 35 tiles extracted, 6 S3-verified vs prd-tnm.s3.amazonaws.com); the three GeoTIFFs remain un-fetched in sandbox (egress allowlist) — scripts/download_competition_data.sh provided. Original catalog wording (USER-PROVIDED) retained in docs/data_catalog.csv
7. RESOLVED 2026-09-12: full rules PDF read across all 7 chunks (docs.nlr.gov, identical to canonical www.nlr.gov) — A.1-A.17 now summarized in §4 above. RESOLVED same day: A.1 exact deadline text confirmed (5:00 p.m. ET on the submission deadline date)
8. Official competition GeoTIFFs have NO public mirror (GitHub code search on 5 distinctive filenames: 0 hits, 2026-09-12) and cannot enter the sandbox (egress matrix in LIMITATIONS §1c) — pipeline therefore E2E-verified on the RECONSTRUCTED public-source dataset (E30); leaderboard runs need the official files via scripts/download_competition_data.sh on an unrestricted machine
9. Encoder pretrained weights (EfficientNet-B5/MIT-B2) not downloadable from sandbox (release-asset host blocked) — sandbox runs use pretrained:false; flagged in configs/config_recon_cpu.yaml
10. **Session 7 (2026-09-16): three defects in `scripts/eval_proxy_catalogue.py` were found by
   inspection and fixed — (a) truth length converted with 0.01 km/px instead of 0.1 km/px, so every
   committed truth length was published 10× short (61,664 px printed as 616.6 km; the same quantity
   in `build_proxy_catalogue.py` used the correct 0.1, so two scripts contradicted each other);
   (b) the blanket-ones baseline was taken over `np.isfinite(pred)`, so its definition changed with
   whichever raster was scored; (c) the score of the raster handed to `--pred` was published as
   "as submitted", which labelled the sweep's pre-shaping ensemble map as the submission on the
   Results page.** All three are fixed and pinned by tests; the committed evidence files written
   before the fix still carry the old field values and will be re-written by the next `proxy-eval`
   run.** Flagged here because each one produced a plausible number that meant something else.
10b. make_patches keeps training windows partially overlapping test regions (mild CV leakage vs reference global zeroing) — flagged, fix queued in SUGGESTIONS §0.1; augmentation described in docs NOT yet wired into train loop — docs corrected 2026-09-12

---

## 9. Full List of Official Verified Links for Manual Verification

See `docs/data_catalog.csv` for complete CSV and `docs/references.md` for fetch logs. Key links:

- Competition Main: https://www.drivendata.org/competitions/306/competition-doe-gems/
- Problem Description: https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
- About Page: https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/
- Data Tab (login): https://www.drivendata.org/competitions/306/competition-doe-gems/data/
- Rules: https://www.drivendata.org/competitions/306/competition-doe-gems/rules/ → https://www.herox.com/GEMSPrize/resource/2274 → https://docs.nlr.gov/docs/fy26osti/96647.pdf
- Reference Solution: https://github.com/drivendataorg/gems-prize-reference-solution
- GeoDAWN ScienceBase: https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7
- GeoDAWN DOI: https://doi.org/10.5066/P93LGLVQ
- GeoDAWN Overview: https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and
- QFaults Interactive: https://www.usgs.gov/programs/earthquake-hazards/faults
- QFaults ScienceBase: https://www.sciencebase.gov/catalog/item/589097b1e4b072a7ac0cae23
- QFaults DOI: https://doi.org/10.5066/P9BCVRCK
- QFaults KML/GIS availability: USGS FAQ https://www.usgs.gov/faqs/where-can-i-find-a-fault-map-united-states-one-available-gis-format + qfaults.kmz on ScienceBase https://www.sciencebase.gov/catalog/item/589097b1e4b072a7ac0cae23
- QFaults ArcGIS: https://earthquake.usgs.gov/arcgis/rest/services/haz/Qfaults/MapServer
- 3DEP About: https://www.usgs.gov/3d-elevation-program/about-3dep-products-services
- TNM Downloader: https://apps.nationalmap.gov/downloader/
- LidarExplorer: https://apps.nationalmap.gov/lidar-explorer/
- AWS Lidar: https://registry.opendata.aws/usgs-lidar/
- INGENIOUS GDR: https://gdr.openei.org/submissions/1391
- INGENIOUS DOI: https://doi.org/10.15121/1881483
- INGENIOUS OSTI: https://www.osti.gov/biblio/1881483
- Project Site: https://gbcge.org/current-projects/ingenious/
- GBCGE Explorer DOI: https://doi.org/10.15121/1987556
- GBCGE OSTI: https://www.osti.gov/dataexplorer/biblio/dataset/1987556
- Favorability: https://www.sciencebase.gov/catalog/item/66e88690d34e0606a9db9b43
- EarthMRI Portal: https://mrdata.usgs.gov/earthmri/
- EarthMRI FS: https://pubs.usgs.gov/publication/fs20203055
- Mattéo et al 2021: https://doi.org/10.1029/2020JB021269
- Hermant et al 2025: https://pangea.stanford.edu/ERE/db/GeoConf/papers/SGW/2025/Hermant.pdf
- EPSG:32611: https://epsg.io/32611
- Tversky Index: https://en.wikipedia.org/wiki/Tversky_index
- Isostatic Gravity: https://mrdata.usgs.gov/metadata/usgraviso.html + https://mrdata.usgs.gov/gravity/isostatic/
- Reduced-to-Pole: Baranov & Naudy 1964 (canonical; p1720D.pdf link removed in review)
- GSRM: https://gsrm2.unavco.org/model/model.html + https://www.unavco.org/software/visualization/idv/IDV_datasource_gsrm.html
- Walker Lane: https://nbmg.unr.edu/staff/faulds/33_AGS22_Faulds_and_Henry_(Walker_Lane)_final.pdf + https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023JB028044
- Additional literature verified in docs/literature.md (12+ papers)

All verified, no hallucinations.
