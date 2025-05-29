import pandas as pd
import re

pd.set_option('display.max_columns', None)

CV_PATH = '/Users/busesomunncu/Desktop/Linkedln Job Prediction Project/cv_dataset.csv'
CATEGORIES_PATH = '/Users/busesomunncu/Desktop/Linkedln Job Prediction Project/f_annotated_categories_and_jobs.csv'


FEATURES = [
    'technical_requirements',
    'soft_skills',
    'domain_knowledge',
    'education_requirement',
    'experience_level',
    'project_experience'
]
WEIGHTS = {
    'technical_requirements': 4,
    'soft_skills': 1,
    'domain_knowledge': 1,
    'education_requirement': 1,
    'experience_level': 2,
    'project_experience': 2
}

def tokenize(text: str):
    return re.findall(r'\b\w+\b', text.lower())

def load_data(cv_path, categories_path):
    jobs_raw = pd.read_csv(categories_path)
    cv_df = pd.read_csv(cv_path)

    # Ensure category_id exists
    if 'category_id' not in jobs_raw.columns:
        jobs_raw.insert(0, 'category_id', range(1, len(jobs_raw) + 1))

    # Merge keywords helper
    def merge_keywords(series):
        all_keys = set()
        for entry in series.dropna():
            for kw in str(entry).split(','):
                key = kw.strip().lower()
                if key:
                    all_keys.add(key)
        return sorted(all_keys)

    # Aggregate rows by category_id
    jobs_df = jobs_raw.groupby('category_id', as_index=False)\
                     .agg({feat: merge_keywords for feat in FEATURES})

    # Assign cv_id
    cv_df.insert(0, 'cv_id', range(1, len(cv_df) + 1))
    return jobs_df, cv_df

def get_user_cv_id():
    return int(input("Enter a CV ID: ").strip())

def calculate_similarity_pct(cv_row, jobs_df):
    cv_keywords = {
        feat: [kw.strip().lower() for kw in str(cv_row[feat]).split(',') if kw.strip()]
        for feat in FEATURES
    }
    records = []
    for _, job_row in jobs_df.iterrows():
        rec = {'category_id': job_row['category_id']}
        for feat in FEATURES:
            if feat in ['domain_knowledge', 'project_experience']:
                job_tokens = set(tokenize(" ".join(job_row[feat])))
                cv_tokens  = set(tokenize(cv_row[feat]))
                matched = job_tokens & cv_tokens
                total = len(job_tokens)
            else:
                job_keys = job_row[feat]
                matched = set(kw for kw in cv_keywords[feat] if kw in job_keys)
                if feat in ['education_requirement', 'experience_level'] and 'unknown' in job_keys:
                    matched.add('unknown')
                total = len(job_keys)
            rec[feat] = round((len(matched)/total*100) if total else 0, 2)
        records.append(rec)
    return pd.DataFrame(records).set_index('category_id')

def main():
    jobs_df, cv_df = load_data(CV_PATH, CATEGORIES_PATH)
    cv_id = get_user_cv_id()
    if cv_id not in cv_df['cv_id'].values:
        print(f"CV ID {cv_id} not found.")
        return

    cv_row = cv_df[cv_df['cv_id'] == cv_id].iloc[0]
    pct_df = calculate_similarity_pct(cv_row, jobs_df)

    print("\n=== Match Percentages ===")
    print(pct_df)

    print("\n=== Top-Match Category for Each Attribute ===")
    for feat in FEATURES:
        best_id = pct_df[feat].idxmax()
        print(f"- {feat}: {best_id} ID'li kategori")

    print("\nWhich criterion is most important for you?")
    for i, feat in enumerate(FEATURES, start=1):
        print(f"{i} - {feat}")
    choice = int(input("Enter your choice (1-6): ").strip())
    chosen = FEATURES[choice-1]

    sorted_df = pct_df.sort_values(by=[chosen, 'technical_requirements'], ascending=[False, False])
    top7 = sorted_df.head(7)

    top7['total_percentage'] = sum(top7[feat] * WEIGHTS[feat] for feat in FEATURES)

    print(f"\n=== Candidate’s Top 7 Categories and Weighted Total (Selected Criterion: {chosen}) ===")
    print(top7)

    top5 = top7.sort_values(by='total_percentage', ascending=False).head(5)
    print("\n=== Recommended 5 Categories (By Weighted Total) ===")
    print(top5[['total_percentage']])

if __name__ == "__main__":
    main()

# main()
