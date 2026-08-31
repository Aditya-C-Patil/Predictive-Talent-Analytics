# **🎯 Predictive Talent Analytics & Semantic Skill-Gap Engine**

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Modeling-Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Imbalanced-Learn](https://img.shields.io/badge/Sampling-SMOTE-green.svg)](https://imbalanced-learn.org/)
[![UI: Streamlit](https://img.shields.io/badge/Interface-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An end-to-end Machine Learning pipeline and interactive analytics platform that vectorizes candidate technical competencies, resolves class imbalance across sparse job domains, and predicts optimal career trajectories while executing curriculum gap analysis.

---

## **📌 Problem Statement & Engineering Challenges**

Traditional keyword-based candidate screening systems struggle with:

1. **Class Imbalance in Role Catalogs:** High-demand roles dominate training splits, causing classical classifiers to bias heavily toward majority classes while failing on niche technical paths.
2. **High-Frequency Term Saturation:** Naive word frequency models over-index on generic terms rather than distinctive technical competencies.
3. **Out-of-Distribution (OOD) Hallucination:** Unconstrained ML models often generate arbitrary baseline predictions when presented with non-technical inputs or missing tokens.

This project addresses these challenges through a combination of sublinear TF-IDF vectorization, SMOTE oversampling, ensemble Random Forest classification, deterministic coverage metrics, and strict vocabulary guardrails.

---

## **🏗️ Architecture & Pipeline Overview**
```text
┌─────────────────────────────────────────────────────────────┐
│                Candidate Competencies Selection             │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│               Sublinear TF-IDF Vectorizer                   │
│   • Unigrams + Bigrams (1, 2)  • Sublinear Scaling 1+log(tf)│
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Vocabulary OOD Guardrail                    │
│       • nnz == 0 ? Halt execution : Forward to Model        │
└──────────────┬───────────────────────────────┬──────────────┘
               │                               │
               ▼                               ▼
┌─────────────────────────────┐   ┌─────────────────────────────┐
│   Random Forest Ensemble    │   │  Fuzzy Skill Gap Resolver   │
│   • 150 Decision Trees      │   │  • Levenshtein Metric (≥0.75│
│   • Probability Calibration │   │  • Role Curriculum Mapping  │
└──────────────┬──────────────┘   └──────────────┬──────────────┘
               │                               │
               ▼                               ▼
      [ Model Likelihood ]            [ Competency Coverage ]
               │                               │
               └───────────────┬───────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│               Streamlit Telemetry & Radar UI                │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 Key Methodologies

### 1. Sublinear Term Frequency Scaling
To prevent high-frequency skills from overwhelming distinctive technical markers, the vectorizer uses sublinear TF scaling:

$$\text{tf}_{\text{sublinear}} = 1 + \log(\text{tf}) \quad \text{for } \text{tf} > 0$$

Combined with bigram extraction ($n \in \{1, 2\}$), the model preserves the semantic identity of compound terms (e.g., separating `Machine Learning` from generic `Learning`).

### 2. Class Imbalance Mitigation (SMOTE)
Candidate dataset distributions often exhibit significant skew between common software roles and niche specializations. The training pipeline applies **Synthetic Minority Over-sampling Technique (SMOTE)** in the sparse feature space:
* Synthesizes convex combinations of neighboring feature vectors ($k=1$).
* Equalizes class priors across all indexed role categories prior to training the ensemble.

### 3. Dual Evaluation Framework: Coverage vs. Likelihood

To deliver transparent analytics, the engine separates **statistical model confidence** from **deterministic curriculum coverage**:

| Metric | Computation | What It Measures |
| :--- | :--- | :--- |
| **Competency Coverage (%)** | $\frac{\text{Matched User Competencies}}{\text{Total Required Role Competencies}} \times 100$ | **Deterministic Overlap:** Exact proportion of the standardized curriculum requirements fulfilled by the candidate. |
| **Model Likelihood (%)** | $P(y = c \mid \vec{x}) = \frac{1}{N_{\text{trees}}} \sum_{i=1}^{N_{\text{trees}}} P_i(y = c \mid \vec{x})$ | **Ensemble Probability:** The percentage of decision trees in the Random Forest ensemble that classified the input feature vector into role $c$. |

* **Example:** If a candidate enters a profile covering 100% of the skills for *Marketing*, *Game Development*, and *Data Science*, each role will reflect **100% Competency Coverage**. However, the **Model Likelihood** will distribute probabilities (e.g., 34.0%, 28.7%, 8.0%) based on the global feature weights and token uniqueness learned during training.

---

## 🛡️ Inference Guardrails

* **Zero-Signal / OOD Protection:** If an input produces a zero-vector (`vec.nnz == 0`), execution stops immediately with a warning rather than evaluating baseline class probabilities.
* **Fuzzy String Alignment:** Skill gap matching combines substring containment with normalized Levenshtein sequence matching ($\ge 0.75$) to reconcile phrasing variations (e.g., matching `HR Management` against `Management`).

---

## 🚀 Installation & Local Deployment

### 1. Clone the Repository
```bash
git clone [https://github.com/Aditya-C-Patil/Predictive-Talent-Analytics.git](https://github.com/Aditya-C-Patil/Predictive-Talent-Analytics.git)
cd Predictive-Talent-Analytics
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Platform
```bash
streamlit run app.py
```
*(On first execution, train.py automatically downloads the dataset, trains the model, caches /models artifacts, and launches the dashboard.)*

## **📂 Repository Structure**
```text
Predictive-Talent-Analytics/
├── app.py              # Modern Streamlit UI, Radar Visualizations & Telemetry
├── train.py            # Data loading, SMOTE resampling, and artifact export
├── models/             # Serialized model, vectorizer, and taxonomy dictionaries
│   ├── classifier.pkl
│   ├── tfidf.pkl
│   ├── role_skills_map.pkl
│   └── all_skills.pkl
├── requirements.txt    # Pinned Python package dependencies
└── README.md           # Engineering documentation
📄 License

This project is licensed under the MIT License — see the LICENSE file for details.
