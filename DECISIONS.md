# 🏛️ Architecture Decision Records (ADR) — Predictive Talent Analytics

This document records the key architectural choices, data engineering trade-offs, and algorithmic decisions made during the design and development of the **Predictive Talent Analytics & Semantic Skill-Gap Engine**.

---

## 📋 Table of Contents
1. [ADR-001: Sublinear TF-IDF with N-Grams vs. Dense Embeddings](#adr-001-sublinear-tfidf-with-n-grams-vs-dense-embeddings)
2. [ADR-002: SMOTE Oversampling in Sparse Feature Spaces](#adr-002-smote-oversampling-in-sparse-feature-spaces)
3. [ADR-003: Random Forest Ensemble vs. Deep Neural Networks](#adr-003-random-forest-ensemble-vs-deep-neural-networks)
4. [ADR-004: Dual-Metric Formulation (Coverage Ratio vs. Model Likelihood)](#adr-004-dual-metric-formulation-coverage-ratio-vs-model-likelihood)
5. [ADR-005: Hybrid Fuzzy String Alignment vs. Rigid Set Difference](#adr-005-hybrid-fuzzy-string-alignment-vs-rigid-set-difference)
6. [ADR-006: Multi-Select Controlled Taxonomy vs. Free-Form Text Entry](#adr-006-multi-select-controlled-taxonomy-vs-free-form-text-entry)

---

### ADR-001: Sublinear TF-IDF with N-Grams vs. Dense Embeddings
* **Status:** Accepted
* **Context:** Candidate skill profiles represent short, sparse collections of technical keywords rather than rich syntactic sentences. Dense semantic sentence transformers (e.g., MiniLM, BERT) introduce heavy inference latency and often produce overly smooth cosine similarities between distinct technical domains (e.g., mapping `Java` to `JavaScript`).
* **Decision:** Selected `TfidfVectorizer` configured with:
  * **N-gram Range $(1, 2)$:** Captures distinct compound entities (e.g., distinguishing `"Machine Learning"` and `"Deep Learning"` from isolated `"Learning"`).
  * **Sublinear Scaling ($1 + \log(\text{tf})$):** Prevents high-frequency generic skills from dominating sparse, specialized domain tokens.
* **Trade-offs:** 
  * *Pros:* Near-instant inference ($< 2\text{ ms}$), zero GPU dependency, interpretable token feature weights.
  * *Cons:* Requires vocabulary alignment for unseen synonyms outside the dataset.

---

### ADR-002: SMOTE Oversampling in Sparse Feature Spaces
* **Status:** Accepted
* **Context:** The underlying candidate dataset contains heavy class imbalance where common web and software roles outnumber specialized tracks (e.g., DevOps, Game Development, ML Engineering). Training without imbalance mitigation led to minority-class starvation and high false-negative rates on specialized paths.
* **Decision:** Implemented **Synthetic Minority Over-sampling Technique (SMOTE)** with $k=1$ nearest neighbors directly within the TF-IDF feature space before model fitting.
* **Trade-offs:**
  * *Pros:* Balances class priors, preventing the classifier from defaulting to majority software roles on ambiguous inputs.
  * *Cons:* Synthesizes convex combinations of sparse feature vectors, slightly increasing training time (mitigated by offline artifact caching in `/models`).

---

### ADR-003: Random Forest Ensemble vs. Deep Neural Networks
* **Status:** Accepted
* **Context:** The system required calibrated multi-class probability outputs across tabular/sparse data that could be serialized and served in lightweight production environments without heavy runtime runtimes (e.g., PyTorch/CUDA).
* **Decision:** Deployed a `RandomForestClassifier` with $n=150$ estimators and parallelized tree execution (`n_jobs=-1`).
* **Trade-offs:**
  * *Pros:* Robust against feature collinearity, non-parametric, natively outputs ensemble vote distributions (`predict_proba`), and serializes to a compact `.pkl` payload.
  * *Cons:* Non-linear probability calibration can be less smooth than logistic regression, resolved by pairing it with deterministic coverage metrics.

---

### ADR-004: Dual-Metric Formulation (Coverage Ratio vs. Model Likelihood)
* **Status:** Accepted
* **Context:** Probabilistic classifiers distribute likelihood across classes, meaning a candidate matching 100% of the skills for three distinct roles will see split likelihoods (e.g., $34\%$, $28\%$, $8\%$). Users often misinterpret this as low qualification.
* **Decision:** Decoupled evaluation into two complementary metrics:
  1. **Competency Coverage ($\%$):** Deterministic ratio $\frac{\text{Matched User Skills}}{\text{Total Required Role Skills}} \times 100$.
  2. **Model Likelihood ($\%$):** Random Forest ensemble voting distribution reflecting global feature uniqueness and dataset priors.
* **Trade-offs:**
  * *Pros:* Clear transparency for end users and recruiters; prevents confusion when candidate profiles span multiple disciplines.

---

### ADR-005: Hybrid Fuzzy String Alignment vs. Rigid Set Difference
* **Status:** Accepted
* **Context:** Standard mathematical set differences (`set(user) - set(role)`) fail on minor naming variations (e.g., matching `"Management"` against `"HR Management"`, or `"SQL"` against `"SQL Database"`).
* **Decision:** Implemented a two-tier matching function combining **substring containment** with a **Levenshtein similarity threshold ($\ge 0.75$)** using Python's `difflib.SequenceMatcher`.
* **Trade-offs:**
  * *Pros:* Prevents false skill-gap alerts for near-identical terms without requiring heavy synonym dictionaries.
  * *Cons:* Minimal computational overhead per comparison, mitigated by running checks only across top-3 predicted roles.

---

### ADR-006: Multi-Select Controlled Taxonomy vs. Free-Form Text Entry
* **Status:** Accepted
* **Context:** Unconstrained free-form text entry invited Out-of-Distribution (OOD) inputs (e.g., non-technical words like `"sports"`), producing zero-signal vectors where the model hallucinated arbitrary class predictions based purely on residual base rates.
* **Decision:** Extracted and deduplicated the entire dataset's technical competency vocabulary into an indexed taxonomy (`all_skills.pkl`) and surfaced it via Streamlit's `st.multiselect` UI component.
* **Trade-offs:**
  * *Pros:* Eliminates zero-vector OOD failures, standardizes syntax, and provides autocomplete search for candidates.
  * *Cons:* Restricts candidate input to the indexed vocabulary catalog.
