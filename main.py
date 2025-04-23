import pandas as pd
import re
import json
import numpy as np
from flashtext import KeywordProcessor
import ast

pd.set_option('display.max_columns', None)


cats = pd.read_csv("/Users/busesomunncu/Desktop/Linkedln Job Prediction Project/categories_with_attributes_updated.csv")
jobs = pd.read_csv("/Users/busesomunncu/Desktop/Linkedln Job Prediction Project/text_mining_data_updated.csv")
df = pd.read_csv("/Users/busesomunncu/Desktop/Linkedln Job Prediction Project/annotated_job_descriptions_fixed.csv")
annotated_df = pd.read_csv("/Users/busesomunncu/Desktop/Linkedln Job Prediction Project/annotated_job_descriptions.csv")

# --------------
def annotate_job_descriptions(
    cats_path: str,
    jobs_path: str,
    output_path: str = "annotated_job_descriptions_fixed.csv"
):
    """
    1) categories CSV ve jobs CSV’ini yükler
    2) Her kategori için her attribute’ta yer alan terimleri (ast.literal_eval + yedek split) alır
    3) FlashText ile her job_description’dan bu terimleri eşleştirir
    4) Eşleşmeleri yeni sütunlar olarak jobs DataFrame’ine ekler ve kaydeder
    """
    # 1) Load data
    cats = pd.read_csv(cats_path)
    jobs = pd.read_csv(jobs_path)

    # 2) Prepare columns
    id_col    = 'category_id'
    skip_cols = {id_col, 'category_name'}
    attr_cols = [c for c in cats.columns if c not in skip_cols]

    # 3) Build a KeywordProcessor per category per attribute
    cat_kps = {}
    for cid, grp in cats.groupby(id_col):
        processors = {}
        for col in attr_cols:
            kp = KeywordProcessor(case_sensitive=False)
            for cell in grp[col].dropna().astype(str):
                try:
                    items = ast.literal_eval(cell)
                except (ValueError, SyntaxError):
                    txt   = cell.strip("[]")
                    items = [t.strip().strip("'\"") for t in txt.split(",") if t.strip()]
                for t in items:
                    term = t.strip().lower()
                    if term:
                        kp.add_keyword(term)
            processors[col] = kp
        cat_kps[cid] = processors

    # 4) Iterate over jobs and extract matches
    results = {col: [] for col in attr_cols}
    for _, row in jobs.iterrows():
        desc       = str(row['job_description']).lower()
        processors = cat_kps.get(row[id_col], {})
        for col in attr_cols:
            kp = processors.get(col)
            if kp:
                hits = set(kp.extract_keywords(desc))
                results[col].append(", ".join(sorted(hits)))
            else:
                results[col].append("")

    # 5) Attach results and save
    for col in attr_cols:
        jobs[col] = results[col]

    jobs.to_csv(output_path, index=False)
    print(f"Annotated file saved as: {output_path}")

# annotate_job_descriptions("categories_with_attributes_updated.csv", "text_mining_data_updated.csv"  output_path="annotated_job_descriptions_fixed.csv" )

# -----------
def prepare_normalized_csv(df):
    """
    Takes an annotated job descriptions DataFrame (with
    'technical_requirements', 'soft_skills', 'domain_knowledge',
    'education_requirement', 'experience_level', 'project_experience')
    and returns a cleaned CSA DataFrame with:
      - Selected key columns
      - Normalized education and experience levels
    """

    # 1) Keep only necessary columns
    keep_cols = [
        'id', 'category_id',
        'technical_requirements', 'soft_skills', 'domain_knowledge',
        'education_requirement', 'experience_level', 'project_experience'
    ]
    df_new = df[keep_cols].copy()

    # 2) Define mappings inside the function
    edu_map = {
        'Bachelor': [
            'bachelor', "bachelor's degree", 'undergraduate', 'university degree', 'university student',
            'college degree', 'b.sc', 'bsc', 'bs degree', 'bachelor of science', 'bachelor of engineering',
            'bachelor of technology', 'beng', 'btech', 'currently studying', 'currently enrolled', 'final year',
            'last year student', '4th grade', '4th year', '3rd grade', '3rd year', 'junior year',
            '2nd year', '2nd grade', '1st year', '1st grade', 'sophomore', 'freshman', 'student internship',
            'student position', 'internship opportunity', 'currently a student', 'ongoing education'
        ],
        'Master': [
            'master', "master's degree", 'm.sc', 'msc', 'ms degree', 'master of science',
            'master of engineering', 'meng', 'postgraduate'
        ],
        'PhD': [
            'phd', 'ph.d', 'doctorate', 'doctoral degree', 'dr.', 'post-doc'
        ]
    }

    exp_map = {
        'Intern': [
            'intern', 'internship', 'student position', 'trainee', 'entry-level internship'
        ],
        'Entry Level': [
            'new grad', 'entry level', 'recent graduate', 'fresh graduate',
            'graduate program', 'no experience required', 'open to graduates'
        ],
        'Junior': [
            '1+ years experience', '1 year experience', '1-2 years',
            'minimum 1 year', 'at least 1 year'
        ],
        'Mid-Level': [
            'mid level', '2+ years experience', '3+ years experience',
            '2-5 years', 'intermediate level', 'at least 3 years',
            'professional experience'
        ],
        'Senior': [
            'senior', '5+ years experience', '6+ years', '5-10 years',
            'experienced', 'minimum 5 years', 'advanced level', 'lead level'
        ],
        'Expert': [
            'team lead', 'technical lead', 'principal engineer',
            'expert level', 'chief', 'staff engineer', 'architect level',
            '10+ years experience'
        ]
    }

    # 3) Define priority lists
    edu_priority = ['PhD', 'Master', 'Bachelor']
    exp_priority = ['Expert', 'Senior', 'Mid-Level', 'Junior', 'Entry Level', 'Intern']

    # 4) Normalization helper
    def normalize_category(cell, mapping, priority, fallback='None'):
        if pd.isna(cell) or not str(cell).strip():
            return fallback
        terms = [t.strip().lower() for t in str(cell).split(',')]
        matched = []
        for std_label, keywords in mapping.items():
            for kw in keywords:
                pattern = rf'.*\b{re.escape(kw)}\b.*'
                if any(re.fullmatch(pattern, t) for t in terms):
                    matched.append(std_label)
                    break
        if not matched:
            return fallback
        if len(matched) == 1:
            return matched[0]
        for label in priority:
            if label in matched:
                return label
        return matched[0]

    # 5) Apply normalization
    df_new['education_requirement'] = df_new['education_requirement'].apply(
        lambda x: normalize_category(x, edu_map, edu_priority)
    )
    df_new['experience_level'] = df_new['experience_level'].apply(
        lambda x: normalize_category(x, exp_map, exp_priority)
    )

    return df_new

# cleaned_df = prepare_normalized_csa(df)
# cleaned_df.to_csv("annotated_job_descriptions_clean.csv", index=False)
# --------- 

def analyze_missing_values(df):
    """
    Önemli sütunlardaki hem gerçek NaN hem de string 'None' değerleri
    sayar, oranını verir ve education_requirement boş/sihirli ilk 5 satırı gösterir.
    """
    cols = [
        'education_requirement',
        'experience_level',
        'technical_requirements',
        'soft_skills',
        'domain_knowledge',
        'project_experience'
    ]
    # hem NaN hem 'None' maskesi
    missing_mask = df[cols].isna() | (df[cols] == 'None')
    
    # say ve oran hesapla
    none_counts = missing_mask.sum().rename("missing_count").to_frame()
    none_counts['total_rows']   = len(df)
    none_counts['pct_missing']  = (none_counts['missing_count'] / len(df) * 100).round(2)

    print("Eksik değer sayıları ve oranları (NaN veya 'None'):\n", none_counts)

    # education_requirement eksik olan ilk 5
    sample = df[missing_mask['education_requirement']] \
               .loc[:, ['id', 'education_requirement']] \
               .head(5)
    print("\nÖrnek satırlar (education_requirement eksik):\n", sample)

df_test = pd.read_csv(
    "/Users/busesomunncu/Desktop/Linkedln Job Prediction Project/annotated_job_descriptions.csv",
    na_values=['None'],
    keep_default_na=True
)

analyze_missing_values(df_test)

# -------------------

def fill_unknown_values(df, cols=None):
    """
    Fills missing (NaN) and explicit 'None' strings with 'Unknown'
    for the specified columns in the DataFrame.
    """
    if cols is None:
        cols = [
            'education_requirement',
            'experience_level',
            'technical_requirements',
            'soft_skills',
            'domain_knowledge',
            'project_experience'
        ]
    for col in cols:
        # Replace 'None' string with NaN, then fill NaN with 'Unknown'
        df[col] = df[col].replace('None', pd.NA).fillna('Unknown')
    return df

new_df = fill_unknown_values(annotated_df)
# new_df.to_csv("/Users/busesomunncu/Desktop/Linkedln Job Prediction Project/annotated_job_descriptions_filled.csv", index=False)

df_test2 = pd.read_csv(
    "/Users/busesomunncu/Desktop/Linkedln Job Prediction Project/annotated_job_descriptions_filled.csv",
    na_values=['None'],
    keep_default_na=True
)

analyze_missing_values(df_test2)

# -----------
