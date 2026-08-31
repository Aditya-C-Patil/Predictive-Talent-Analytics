# 🎯 Predictive Talent Analytics & Semantic Skill-Gap Engine

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Modeling-Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Imbalanced-Learn](https://img.shields.io/badge/Sampling-SMOTE-green.svg)](https://imbalanced-learn.org/)
[![UI: Streamlit](https://img.shields.io/badge/Interface-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An end-to-end Machine Learning pipeline and modern analytics platform that vectorizes candidate technical competencies, mitigates class imbalance across specialized career paths, and predicts optimal job trajectories while performing granular curriculum gap analysis.

---

## 📌 Problem Statement & Engineering Challenges

Traditional keyword-matching career advisory tools suffer from fundamental flaws:
1. **Severe Class Imbalance:** Common job roles dominate tabular candidate datasets, causing standard classifiers to underperform on specialized career tracks.
2. **High-Frequency Term Saturation:** Naive frequency-based representations over-weight generic words rather than distinctive technical competencies.
3. **Fragile Exact Matching:** Rigid string-equality algorithms fail when candidate inputs contain minor syntax or naming variations (e.g., `Management` vs. `HR Management`).

This engine addresses these challenges through a combination of **sublinear TF-IDF n-gram vectorization**, **SMOTE oversampling**, **ensemble Random Forest classification**, **dual-metric evaluation**, and **fuzzy Levenshtein gap alignment**.

---

## 🏗️ Architecture & Pipeline
```text
┌─────────────────────────────────────────────────────────────┐
│           Candidate Competency Taxonomy Selector            │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│               Sublinear TF-IDF Vectorizer                   │
│   • Unigrams + Bigrams (1, 2)  • Sublinear Scaling 1+log(tf)│
└──────────────────────────────┬──────────────────────────────┘
                               │
                  ┌────────────┴────────────┐
                  ▼                         ▼
     ┌─────────────────────────┐ ┌─────────────────────────────┐
     │   Random Forest Ensemble│ │  Fuzzy Skill Gap Resolver   │
     │   • 150 Decision Trees   │ │  • Levenshtein Distance     │
     │   • Probability Calibration│ │  • Target Role Mapping    │
     └────────────┬────────────┘ └──────────────┬──────────────┘
                  │                             │
                  ▼                             ▼
       [ Model Likelihood % ]        [ Competency Coverage % ]
                  │                             │
                  └─────────────┬───────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│               Glassmorphic Streamlit Dashboard              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 Core Methodologies

### 1. Sublinear Term Frequency Scaling
To prevent high-frequency skills from skewing model decisions, the vectorizer incorporates sublinear TF scaling:

$$\text{tf}_{\text{sublinear}} = 1 + \log(\text{tf}) \quad \text{for } \text{tf} > 0$$

Paired with bigram extraction ($n \in \{1, 2\}$), the feature extractor distinguishes specialized compound terms (such as `Machine Learning`) from isolated baseline tokens.

### 2. Class Imbalance Mitigation (SMOTE)
To ensure equitable representation across sparse roles in the dataset, the training pipeline executes **Synthetic Minority Over-sampling Technique (SMOTE)** in the sparse feature space:
* Synthesizes convex interpolations between neighboring vectors ($k=1$).
* Balances prior distributions across all indexed job categories before training.

### 3. Dual Evaluation Framework: Coverage vs. Likelihood

The system cleanly distinguishes **statistical ensemble confidence** from **deterministic curriculum coverage**:

| Metric | Calculation | Definition |
| :--- | :--- | :--- |
| **Competency Coverage (%)** | $\frac{\text{Matched User Competencies}}{\text{Total Required Role Competencies}} \times 100$ | **Deterministic Overlap:** Exact percentage of standard curriculum skills possessed by the candidate for that target role. |
| **Model Likelihood (%)** | $P(y = c \mid \vec{x}) = \frac{1}{N_{\text{trees}}} \sum_{i=1}^{N_{\text{trees}}} P_i(y = c \mid \vec{x})$ | **Ensemble Probability:** The percentage of decision trees ($n=150$) that classified the input feature vector into role $c$ based on global TF-IDF weights. |

* **Example:** If a candidate enters a profile fulfilling 100% of the skills for *Marketing*, *Game Developer*, and *Data Scientist*, each role correctly displays **100% Competency Coverage**. Concurrently, the **Model Likelihood** divides the 150 decision tree votes (e.g., 34.0%, 28.7%, 8.0%) based on token uniqueness and learned feature importances.

### 4. Fuzzy Semantic String Alignment
To prevent missing valid sub-competencies, the gap engine pairs substring matching with normalized Levenshtein sequence matching ($\text{ratio} \ge 0.75$), correctly associating terms like `HR Management` with `Management`.

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
