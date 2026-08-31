import os
import pickle
import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

def train_and_export():
    print("📥 Loading candidate dataset...")
    path = kagglehub.dataset_download("ckshetty/candidate-job-role-dataset")
    csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
    df = pd.read_csv(os.path.join(path, csv_file))

    # Normalize text
    df['skills'] = df['skills'].fillna("").str.lower().str.replace(r"[\[\]']", "", regex=True)
    counts = df['job_role'].value_counts()
    df = df[df['job_role'].isin(counts[counts >= 5].index)].copy()

    # Sublinear TF-IDF
    tfidf = TfidfVectorizer(stop_words='english', max_features=4000, ngram_range=(1, 2), sublinear_tf=True)
    X = tfidf.fit_transform(df['skills'])
    y = df['job_role']

    # SMOTE balancing
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_res, y_res = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print(f"✅ Trained Model Accuracy: {accuracy_score(y_test, model.predict(X_test))*100:.2f}%")

    os.makedirs("models", exist_ok=True)
    with open("models/classifier.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)

    # Build unique role-to-skills lookup and unique master skill catalog
    role_skills_map = {}
    all_unique_skills = set()

    for role, group in df.groupby('job_role'):
        combined_text = group['skills'].str.cat(sep=', ')
        parsed = sorted(list(set([s.strip().lower() for s in combined_text.split(',') if len(s.strip()) > 1])))
        role_skills_map[role] = parsed
        all_unique_skills.update(parsed)

    with open("models/role_skills_map.pkl", "wb") as f:
        pickle.dump(role_skills_map, f)

    with open("models/all_skills.pkl", "wb") as f:
        pickle.dump(sorted(list(all_unique_skills)), f)

    print("💾 Model and skill catalog exported to /models.")

if __name__ == "__main__":
    train_and_export()