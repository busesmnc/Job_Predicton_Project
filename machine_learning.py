from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter
import pandas as pd
import itertools

# 1. Etiketli veriyi yükle
labeled_df = pd.read_csv("/Users/busesomunncu/Desktop/Linkedln Job Prediction Project/final_labeled-data2.csv")
all_jobs_df = pd.read_csv("/Users/busesomunncu/Desktop/Linkedln Job Prediction Project/text_mining_data_updated.csv")

# Ayarlar
threshold = 0.3
max_features = 5000
random_state = 42

# Ortak TF-IDF dönüşüm fonksiyonu
def get_tfidf(texts):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    return vectorizer.fit_transform(texts), vectorizer

# Ortak multilabel eğitim fonksiyonu
def multilabel_training(X_text, y_raw, label_list, unlabeled_text):
    y_filtered = y_raw.apply(lambda tags: [tag for tag in tags if tag in label_list])
    non_empty = y_filtered.apply(lambda x: len(x) > 0)
    X_filtered = X_text[non_empty]
    y_filtered = y_filtered[non_empty]
    X_tfidf, tfidf_vectorizer = get_tfidf(X_filtered)
    mlb = MultiLabelBinarizer(classes=label_list)
    y_binary = mlb.fit_transform(y_filtered)
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_binary, test_size=0.2, random_state=random_state)
    model = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=random_state))
    model.fit(X_train, y_train)
    X_unlabeled = tfidf_vectorizer.transform(unlabeled_text)
    proba = model.predict_proba(X_unlabeled)
    preds = mlb.inverse_transform((proba >= threshold).astype(int))
    return ["; ".join(p) if p else "" for p in preds]

# Tekli sınıf tahmin fonksiyonu (experience, education)
def single_label_training(X_text, y_labels, unlabeled_text):
    X_tfidf, tfidf_vectorizer = get_tfidf(X_text)
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_labels, test_size=0.2, random_state=random_state)
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    X_unlabeled = tfidf_vectorizer.transform(unlabeled_text)
    preds = model.predict(X_unlabeled)
    return preds

# --- tech_requirements ---
tech = labeled_df.dropna(subset=["tech_requirements"])
tech_y = tech["tech_requirements"].apply(lambda x: [t.strip().lower() for t in x.split(",") if t.strip()])
top_tech = [t for t, _ in Counter(itertools.chain.from_iterable(tech_y)).most_common(20)]
all_jobs_df["predicted_tech_requirements"] = multilabel_training(tech["job_description"].fillna(""), tech_y, top_tech, all_jobs_df["job_description"].fillna(""))

# --- soft_skills ---
soft = labeled_df.dropna(subset=["soft_skills"])
soft_y = soft["soft_skills"].apply(lambda x: [t.strip().lower() for t in x.split(",") if t.strip()])
top_soft = [t for t, _ in Counter(itertools.chain.from_iterable(soft_y)).most_common(30)]
all_jobs_df["predicted_soft_skills"] = multilabel_training(soft["job_description"].fillna(""), soft_y, top_soft, all_jobs_df["job_description"].fillna(""))

# --- experience_level ---
exp = labeled_df.dropna(subset=["experience_level"])
exp_map = {
    "intern": "intern", "new grad": "entry", "entry": "entry",
    "junior": "junior", "mid": "mid", "midlevel": "mid",
    "midsenior": "mid", "senior": "senior"
}
exp_y = exp["experience_level"].str.lower().str.strip().map(exp_map).dropna()
all_jobs_df["predicted_experience_level"] = single_label_training(exp.loc[exp_y.index, "job_description"].fillna(""), exp_y, all_jobs_df["job_description"].fillna(""))

# --- education_requirement ---
edu = labeled_df.dropna(subset=["education_requirement"])
edu_map = {
    "bachelor": "bachelor", "undergraduate": "bachelor",
    "high school": "high school", "bachelor, master,  phd": "phd"
}
edu_y = edu["education_requirement"].str.lower().str.strip().map(edu_map).dropna()
edu_binary = edu_y.apply(lambda x: "bachelor" if x == "bachelor" else "other")
all_jobs_df["predicted_education_requirement"] = single_label_training(edu.loc[edu_binary.index, "job_description"].fillna(""), edu_binary, all_jobs_df["job_description"].fillna(""))

# --- domain_knowledge ---
domain = labeled_df.dropna(subset=["domain_knowledge"])
domain_y = domain["domain_knowledge"].apply(lambda x: [t.strip().lower() for t in x.split(",") if t.strip()])
top_domain = [t for t, _ in Counter(itertools.chain.from_iterable(domain_y)).most_common(30)]
all_jobs_df["predicted_domain_knowledge"] = multilabel_training(domain["job_description"].fillna(""), domain_y, top_domain, all_jobs_df["job_description"].fillna(""))

# --- project_experience ---
proj = labeled_df.dropna(subset=["project_experience"])
proj_y = proj["project_experience"].apply(lambda x: [t.strip().lower() for t in x.split(",") if t.strip()])
top_proj = [t for t, _ in Counter(itertools.chain.from_iterable(proj_y)).most_common(30)]
all_jobs_df["predicted_project_experience"] = multilabel_training(proj["job_description"].fillna(""), proj_y, top_proj, all_jobs_df["job_description"].fillna(""))

# --- certifications ---
cert = labeled_df.dropna(subset=["certifications"])
cert_y = cert["certifications"].apply(lambda x: [t.strip().lower() for t in x.split(",") if t.strip()])
top_cert = [t for t, _ in Counter(itertools.chain.from_iterable(cert_y)).most_common(30)]
all_jobs_df["predicted_certifications"] = multilabel_training(cert["job_description"].fillna(""), cert_y, top_cert, all_jobs_df["job_description"].fillna(""))

# Final CSV
all_jobs_df.to_csv("/Users/busesomunncu/Desktop/Linkedln Job Prediction Project/final_enriched_dataset.csv", index=False)
print("Final enriched dataset saved as final_enriched_dataset.csv")
