# Machine Learning Hands-On Datasets & Latest Research Benchmarks (2025–2026)

A curated reference guide combining **large-scale Kaggle datasets** for hands-on machine learning practice and **recent ML research projects, competitions, and benchmarks (2025–2026)** with their problem statements and official links.

---

## Table of Contents

1. [Large Kaggle Datasets for Hands-On ML](#1-large-kaggle-datasets-for-hands-on-ml)
   - [Tabular, Fraud & Time-Series](#tabular-fraud--time-series)
   - [Recommender Systems](#recommender-systems)
   - [Natural Language Processing (NLP)](#natural-language-processing-nlp)
   - [Computer Vision & Multimodal](#computer-vision--multimodal)
   - [Kaggle CLI Setup & Best Practices](#kaggle-cli-setup--best-practices)
2. [Latest ML Research Projects & Benchmarks (2025–2026)](#2-latest-ml-research-projects--benchmarks-20252026)
   - [Kaggle Research Competitions (2026 Cycle)](#kaggle-research-competitions-2026-cycle)
   - [Open Research Benchmarks & Datasets](#open-research-benchmarks--datasets)

---

## 1. Large Kaggle Datasets for Hands-On ML

Practicing on large datasets helps build real-world engineering skills around memory optimization, out-of-core pipelines (`Polars`, `DuckDB`, `PySpark`, Pandas `chunksize`), feature engineering at scale, and handling class imbalance.

### Tabular, Fraud & Time-Series

| Dataset / Challenge | Scale & Focus | Key Skills Practiced | Link / CLI Command |
| :--- | :--- | :--- | :--- |
| **IEEE-CIS Fraud Detection** | ~1 GB+ transactional & identity tables | Imbalanced classification, feature engineering, numeric downcasting, LightGBM/XGBoost | `kaggle competitions download -c ieee-fraud-detection` |
| **NYC Yellow Taxi Trip Data** | Multi-million row trip records | Fare/duration regression, spatial & temporal features, large CSV handling | [Kaggle Dataset](https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data) (`kaggle datasets download -d elemento/nyc-yellow-taxi-trip-data`) |
| **Credit Card Fraud Detection** | Anonymized PCA features | Anomaly detection, severe class imbalance, precision-recall evaluation | [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (`kaggle datasets download -d mlg-ulb/creditcardfraud`) |
| **Store Sales – Time Series Forecasting** | Millions of retail time-series rows | Time-series CV, lag & rolling-window features, multi-store seasonality | Search on [Kaggle Competitions](https://www.kaggle.com/competitions) |

### Recommender Systems

| Dataset / Challenge | Scale & Focus | Key Skills Practiced | Link / CLI Command |
| :--- | :--- | :--- | :--- |
| **H&M Personalized Fashion Recommendations** | 10+ GB (transactions, metadata, images) | Candidate generation + two-stage ranking, implicit feedback | `kaggle competitions download -c h-and-m-personalized-fashion-recommendations` |
| **Netflix Prize Data** | 100M+ user-movie ratings | Collaborative filtering, SVD/ALS, sparse matrix operations | [Kaggle Dataset](https://www.kaggle.com/datasets/netflix/netflix-prize-data) (`kaggle datasets download -d netflix/netflix-prize-data`) |

### Natural Language Processing (NLP)

| Dataset / Challenge | Scale & Focus | Key Skills Practiced | Link / CLI Command |
| :--- | :--- | :--- | :--- |
| **Jigsaw Toxic Comment Classification** | Hundreds of thousands of comments | Multi-label text classification, noisy text, TF-IDF baselines to Transformer fine-tuning | `kaggle competitions download -c jigsaw-toxic-comment-classification-challenge` |
| **Amazon Reviews 2018 (Full Dataset)** | Large multi-category review corpus | Sentiment modeling, text embeddings, scalable NLP pipelines | [Kaggle Dataset](https://www.kaggle.com/datasets/skillsmuggler/amazon-reviews-2018) (`kaggle datasets download -d skillsmuggler/amazon-reviews-2018`) |
| **Feedback Prize – Evaluating Student Writing** | Large annotated student essay corpus | Span/token classification, long-sequence modeling, offset alignment | Search on [Kaggle Competitions](https://www.kaggle.com/competitions) |

### Computer Vision & Multimodal

| Dataset / Challenge | Focus | Key Skills Practiced |
| :--- | :--- | :--- |
| **PetFinder.my – Pawpularity Contest** | Multimodal (pet photos + tabular metadata) | Combining CNN/ViT image backbones with structured tabular features |
| **Plant Pathology 2021 – FGVC8** | Multi-GB high-resolution leaf images | Fine-grained image classification, custom PyTorch `Dataset`/`DataLoader` pipelines |
| **RSNA Intracranial Hemorrhage Detection** | Medical imaging (DICOM scans) | Handling DICOM format, 2D/3D CNNs, data augmentation under class imbalance |

### Kaggle CLI Setup & Best Practices

1. **Install the Kaggle CLI:**
   ```bash
   pip install --user kaggle
   ```
2. **Configure API credentials:**
   - Go to **Kaggle → Account → Settings → API → Create New Token** to download `kaggle.json`.
   - Place it at `~/.kaggle/kaggle.json` with restricted permissions:
     ```bash
     mkdir -p ~/.kaggle
     chmod 600 ~/.kaggle/kaggle.json
     ```
3. **Download & unzip datasets:**
   ```bash
   kaggle datasets download -d <owner/dataset-slug> --unzip
   ```
   *(For competition datasets using `kaggle competitions download -c <competition-slug>`, accept the competition rules on Kaggle first.)*
4. **Memory & storage tips for large files:**
   - Downcast numeric columns (`float64` → `float32`, `int64` → `int32`/`int16`) to cut memory footprint.
   - Use **Polars** or **DuckDB** for lazy/out-of-core queries instead of loading multi-GB CSVs eagerly into Pandas.
   - Use Kaggle Notebooks / Colab / Cloud Storage when working with 10+ GB datasets in disk-constrained environments.

---

## 2. Latest ML Research Projects & Benchmarks (2025–2026)

### Kaggle Research Competitions (2026 Cycle)

#### 1. Smart Buildings Competition (NeurIPS 2026)
- **Domain:** Reinforcement Learning & Control / Climate AI
- **Problem Statement:** Train an HVAC control policy using simulated building datasets (varying sizes, layouts, weather files, materials, and HVAC configurations) that generalizes and adapts to **entirely unseen buildings** with minimal setup—addressing the cross-building transferability bottleneck for global energy savings.
- **Link:** [https://www.kaggle.com/competitions/neurips-2026-smart-buildings](https://www.kaggle.com/competitions/neurips-2026-smart-buildings)

#### 2. Solar Filament Segmentation Challenge 2026
- **Domain:** Computer Vision / Heliophysics
- **Problem Statement:** Develop image segmentation algorithms to automatically detect and delineate solar filaments (e.g., from GONG H-Alpha observations), producing pixel-level masks that capture full filament morphologies and fine-scale structures while avoiding non-filament interference.
- **Link:** [https://www.kaggle.com/competitions/filament-segmentation-2026](https://www.kaggle.com/competitions/filament-segmentation-2026)

#### 3. Enveda CASMI 2026 – Molecule ID From Mass Spectra
- **Domain:** AI for Science / Cheminformatics & Metabolomics
- **Problem Statement:** Predict the **2D chemical structures** of small molecules directly from mass spectrometry data acquired from complex biological extracts.
- **Link:** [https://www.kaggle.com/competitions/enveda-casmi-2026](https://www.kaggle.com/competitions/enveda-casmi-2026)

---

### Open Research Benchmarks & Datasets

#### 1. ResearchCodeBench
- **Domain:** LLMs for Automated ML Research Engineering
- **Problem Statement:** Evaluates LLMs on implementing **novel, end-to-end machine learning research code** directly from papers rather than isolated code snippets.
- **Links:**
  - Project Page: [https://researchcodebench.github.io/](https://researchcodebench.github.io/)
  - Paper: [https://huggingface.co/papers/2506.02314](https://huggingface.co/papers/2506.02314)

#### 2. FGBench
- **Domain:** LLMs for Chemistry & Drug Discovery
- **Problem Statement:** Evaluates **functional group-level molecular property reasoning** in Large Language Models to measure how well models reason over chemical substructures and molecular properties.
- **Links:**
  - GitHub: [https://github.com/xuanliugit/FGBench](https://github.com/xuanliugit/FGBench)
  - Hugging Face Dataset: [https://huggingface.co/datasets/xuan-liu/FGBench](https://huggingface.co/datasets/xuan-liu/FGBench)

#### 3. SWE-bench (Verified / Lite / Multimodal)
- **Domain:** Autonomous Software Engineering Agents
- **Problem Statement:** Benchmarks whether AI coding agents can resolve **real-world GitHub issues** across full repositories—requiring multi-file navigation, patch synthesis, execution-verified unit tests, and visual/UI debugging in the Multimodal subset.
- **Links:**
  - GitHub: [https://github.com/swe-bench/SWE-bench](https://github.com/swe-bench/SWE-bench)
  - Hugging Face Collection: [https://huggingface.co/collections/SWE-bench/swe-bench-68113bc99eb3a64a91ea33c9](https://huggingface.co/collections/SWE-bench/swe-bench-68113bc99eb3a64a91ea33c9)

#### 4. ARC-AGI & ARC-AGI-2
- **Domain:** Abstract Reasoning & Generalization
- **Problem Statement:** Measures **fluid intelligence and skill acquisition** on novel grid-transformation tasks where memorization fails, requiring models or program synthesis systems to infer underlying rules from just a few input-output demonstrations.
- **Links:**
  - GitHub (v1): [https://github.com/fchollet/ARC-AGI](https://github.com/fchollet/ARC-AGI)
  - GitHub (v2): [https://github.com/arcprize/ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2)
  - Hugging Face Dataset Mirror: [https://huggingface.co/datasets/dataartist/arc-agi](https://huggingface.co/datasets/dataartist/arc-agi)
