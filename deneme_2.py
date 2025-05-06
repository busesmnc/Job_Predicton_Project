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

# --- 3. Eşleşmeyi hesaplayacak yardımcı fonksiyon ---
def top_matches(raw_value, cat_sets, tokenize=False, top_n=3):
    text = str(raw_value).lower()
    if tokenize:
        cv_items = set(re.findall(r'\w+', text))
    else:
        cv_items = set(s.strip() for s in text.split(',') if s.strip())
    results = []
    for cid, items in cat_sets.items():
        pct = (len(cv_items & items) / len(cv_items) * 100) if cv_items else 0
        results.append((cid, pct))
    results.sort(key=lambda x: (-x[1], x[0]))
    return results[:top_n]

# --- 4. İş ilanları için sütun set’lerini hazırla ---
tech_sets   = build_category_sets(job_df, 'technical_requirements', tokenize=False)
soft_sets   = build_category_sets(job_df, 'soft_skills',              tokenize=False)
domain_sets = build_category_sets(job_df, 'domain_knowledge',        tokenize=True)
exp_sets    = build_category_sets(job_df, 'experience_level',         tokenize=False)
proj_sets   = build_category_sets(job_df, 'project_experience',       tokenize=True)
edu_sets    = build_category_sets(job_df, 'education_requirement',    tokenize=False)

# --- 5. İnteraktif fonksiyon ---
def analyze_cv_custom():
    # CV seçimi
    try:
        sel = int(input(f"İncelemek istediğin CV numarasını gir (1-{len(cv_df)}): "))
        if not (1 <= sel <= len(cv_df)):
            raise ValueError
    except ValueError:
        print("Geçersiz numara. Çıkılıyor.")
        sys.exit(1)

    cv = cv_df.iloc[sel-1]
    name = cv['cv_name']

    # 5a. Her kriter için top 3 eşleşmeyi hesapla
    criteria = [
        ('technical',   cv['technical_requirements'], tech_sets,   False),
        ('soft',        cv['soft_skills'],              soft_sets,   False),
        ('domain',      cv['domain_knowledge'],         domain_sets, True),
        ('experience',  cv['experience_level'],         exp_sets,    False),
        ('project',     cv['project_experience'],       proj_sets,   True),
        ('education',   cv['education_requirement'],    edu_sets,    False),
    ]

    # Top3 id listeleri ve yüzdeler
    top3 = {}
    for crit, raw, sets_, tok in criteria:
        top3[crit] = top_matches(raw, sets_, tokenize=tok, top_n=3)

    # 5b. Detailed matches tablosunu oluştur
    rows = []
    for i in range(3):
        row = {'cv_name': name}
        for crit in ['technical', 'soft', 'domain', 'experience', 'project', 'education']:
            cid, pct = top3[crit][i]
            row[crit] = f"{cid} - {pct:.1f}%"
        rows.append(row)
    df_out = pd.DataFrame(rows, columns=['cv_name','technical','soft','domain','experience','project','education'])
    df_out.index = [name] * len(df_out)

    # 5c. Çıktıları yazdır
    print(f"\n--- Detailed matches for {name} ---")
    print(df_out.to_string())

    # 5d. Sum_pct ve vote_count önerileri
    # sum_pct
    sum_scores = {cid: sum([pct for crit in top3 for cid_, pct in [top3[crit][j] for j in range(3)] if cid_ == cid])
                  for cid in {cid for crit in top3 for cid, _ in top3[crit]}}
    best_sum = max(sum_scores, key=sum_scores.get)
    print(f"\nsum_pct: Tüm kriter yüzdelerinin toplamına göre sana en uygun kategori → {best_sum}")
    # vote_count
    vote_counts = {cid: sum(1 for crit in top3 if any(cid == cid_ for cid_, _ in top3[crit])) 
                   for cid in sum_scores}
    best_vote = max(vote_counts, key=vote_counts.get)
    print(f"vote_count: Her kriterde top 3’e girme sayısına göre sana en uygun kategori → {best_vote}")

    # 5e. Kullanıcının önemli kriter seçimi
    print("\nSenin için en önemli kriter ne?")
    for idx, crit in enumerate(['technical','soft','domain','experience','project','education'], start=1):
        print(f"{idx}: {crit.capitalize()}")
    choice = input("Seçimin (1-6): ").strip()
    try:
        idx = int(choice) - 1
        crit = ['technical','soft','domain','experience','project','education'][idx]
    except:
        print("Geçersiz seçim.")
        return

    # 5f. Seçilen kritere göre top 3 yazdır
    print(f"\n{crit.capitalize()} kriterine göre en uygun 3 kategori:")
    for cid, pct in top3[crit]:
        print(f"  • {cid} → %{pct:.1f}")

# --- 6. Doğrudan çağırmak için ---
if __name__ == "__main__":
    analyze_cv_custom()

# Kullanıcı: analyze_cv_custom() fonksiyonunu çağırman yeterli.

# kategoriler id'ye göre sıralanıyor..