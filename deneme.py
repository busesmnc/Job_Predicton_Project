import pandas as pd
import re
import sys

pd.set_option('display.max_columns', None)

# --- 1. CSV'leri yükle (dışarda kalsın) ---
cv_df  = pd.read_csv('cv_dataset.csv')
job_df = pd.read_csv('annotated_job_descriptions_filled.csv')

# --- 2. Kategoriler için set’ler oluşturacak fonksiyon ---
def build_category_sets(df, col, tokenize=False):
    cat_sets = {}
    for _, row in df.iterrows():
        cid = row['category_id']
        raw = str(row[col]).lower()
        if tokenize:
            items = set(re.findall(r'\w+', raw))
        else:
            items = set(s.strip() for s in raw.split(',') if s.strip())
        cat_sets.setdefault(cid, set()).update(items)
    return cat_sets

# --- 3. Eşleşmeyi hesaplayacak yardımcı fonksiyonu ---
def top_matches(raw_value, cat_sets, tokenize=False):
    text = str(raw_value).lower()
    if tokenize:
        cv_items = set(re.findall(r'\w+', text))
    else:
        cv_items = set(s.strip() for s in text.split(',') if s.strip())
    results = []
    for cid, items in cat_sets.items():
        inter = cv_items & items
        pct   = (len(inter) / len(cv_items) * 100) if cv_items else 0
        results.append((cid, pct, inter))
    results.sort(key=lambda x: (-x[1], x[0]))
    return results[:3]

# --- 4. İş ilanları için sütun set’lerini hazırla ---
tech_sets   = build_category_sets(job_df, 'technical_requirements', tokenize=False)
soft_sets   = build_category_sets(job_df, 'soft_skills',              tokenize=False)
domain_sets = build_category_sets(job_df, 'domain_knowledge',        tokenize=True)
exp_sets    = build_category_sets(job_df, 'experience_level',         tokenize=False)
proj_sets   = build_category_sets(job_df, 'project_experience',       tokenize=True)
edu_sets    = build_category_sets(job_df, 'education_requirement',    tokenize=False)

# --- 5. Tüm interaktif akışı bir fonksiyona taşı ---
def analyze_cv():
    # 5a. Hangi CV'yi seçtiğimizi sor
    try:
        sel = int(input(f"İncelemek istediğin CV numarasını gir (1-{len(cv_df)}): "))
        if not (1 <= sel <= len(cv_df)):
            raise ValueError
    except ValueError:
        print("Geçersiz numara. Çıkılıyor.")
        sys.exit(1)

    cv = cv_df.iloc[sel-1]
    name = cv['cv_name']

    # 5b. Simple print’ler
    print(f"\n=== {name} için eşleşmeler ===")
    print("Technical Requirements:", top_matches(cv['technical_requirements'], tech_sets,   tokenize=False))
    print("Soft Skills:           ", top_matches(cv['soft_skills'],              soft_sets,   tokenize=False))
    print("Domain Knowledge:      ", top_matches(cv['domain_knowledge'],        domain_sets, tokenize=True))
    print("Experience Level:      ", top_matches(cv['experience_level'],         exp_sets,    tokenize=False))
    print("Project Experience:    ", top_matches(cv['project_experience'],       proj_sets,   tokenize=True))
    print("Education Requirement: ", top_matches(cv['education_requirement'],    edu_sets,    tokenize=False))

    # 5c. Detaylı tablo şeklinde göster
    tech_top  = top_matches(cv['technical_requirements'], tech_sets,   tokenize=False)
    soft_top  = top_matches(cv['soft_skills'],              soft_sets,   tokenize=False)
    dom_top   = top_matches(cv['domain_knowledge'],        domain_sets, tokenize=True)
    exp_top   = top_matches(cv['experience_level'],         exp_sets,    tokenize=False)
    proj_top  = top_matches(cv['project_experience'],       proj_sets,   tokenize=True)
    edu_top   = top_matches(cv['education_requirement'],    edu_sets,    tokenize=False)

    rows = []
    for i in range(3):
        rows.append({
            'cv_name':    name,
            'technical':  f"{tech_top[i][0]} - {tech_top[i][1]:.1f}%",
            'soft':       f"{soft_top[i][0]} - {soft_top[i][1]:.1f}%",
            'domain':     f"{dom_top[i][0]} - {dom_top[i][1]:.1f}%",
            'experience': f"{exp_top[i][0]} - {exp_top[i][1]:.1f}%",
            'project':    f"{proj_top[i][0]} - {proj_top[i][1]:.1f}%",
            'education':  f"{edu_top[i][0]} - {edu_top[i][1]:.1f}%"
        })
    df_out = pd.DataFrame(rows, columns=[
        'cv_name','technical','soft','domain','experience','project','education'
    ])
    # Index olarak CV adı göster
    df_out.index = [name] * len(df_out)

    print(f"\n--- Detailed matches for {name} ---")
    print(df_out.to_string())

    # 5d. Öneri: kategori puanlaması
    scores = {}
    for col, sets, tok in [
        ('technical_requirements', tech_sets, False),
        ('soft_skills',            soft_sets, False),
        ('domain_knowledge',       domain_sets, True),
        ('experience_level',       exp_sets,   False),
        ('project_experience',     proj_sets,  True),
        ('education_requirement',  edu_sets,   False),
    ]:
        for cid, pct, _ in top_matches(cv[col], sets, tokenize=tok):
            scores[cid] = scores.get(cid, 0) + pct

    recommended = sorted(scores.items(), key=lambda x: -x[1])[:3]
    print("\nÇıktıya göre sana en uygun üç kategori şunlar gözüküyor:",
          ", ".join(f"{cid} (%{score:.1f})" for cid, score in recommended))

    # 5e. Kullanıcıya kriter önceliği sor
    print("\nHangi kritere öncelik vermek istersin?")
    print("1: Technical Requirements\n2: Soft Skills\n3: Domain Knowledge")
    print("4: Experience Level\n5: Project Experience\n6: Education Requirement")
    choice = input("Seçimin (1-6): ").strip()

    mapping = {
        '1': ('technical_requirements', tech_sets, False, "Technical Requirements"),
        '2': ('soft_skills',            soft_sets, False, "Soft Skills"),
        '3': ('domain_knowledge',       domain_sets, True,  "Domain Knowledge"),
        '4': ('experience_level',       exp_sets,    False, "Experience Level"),
        '5': ('project_experience',     proj_sets,   True,  "Project Experience"),
        '6': ('education_requirement',  edu_sets,    False, "Education Requirement"),
    }

    if choice in mapping:
        col, sets, tok, label = mapping[choice]
        top3 = top_matches(cv[col], sets, tokenize=tok)
        print(f"\nSeçtiğin kriter ({label}) için en iyi 3 kategori:")
        for cid, pct, inter in top3:
            print(f"  • {cid} → %{pct:.1f} (ortak: {inter})")
    else:
        print("Geçersiz seçim. Program sonlanıyor.")

# --- 6. Script doğrudan çalıştırıldığında fonksiyonu çağır ---
if __name__ == "__main__":
    analyze_cv()

# Kullanıcı: analize_cv() fonksiyonunu çağırmanız yeterlidir.

