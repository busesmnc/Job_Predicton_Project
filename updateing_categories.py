import pandas as pd

# 1) Old→new mapping
old_to_new = {
    1:  1,   2:  1,
    4:  2,  16:  2,
    6:  3,  12:  3,   8:  3,
   24:  4,  29:  4,
   18:  5,  19:  5,  21:  5,
    3:  6,
    5:  7,
    7:  8,
    9:  9,
   10: 10,
   11: 11,
   13: 12,
   14: 13,
   15: 14,
   17: 15,
   20: 16,
   22: 17,
   23: 18,
   25: 19,
   26: 20,
   27: 21,
   28: 22
}

df = pd.read_csv('/Users/busesomunncu/Desktop/Linkedln Job Prediction Project/final_annotated_categories_and_jobs.csv')

df['category_id'] = df['category_id'].map(old_to_new)

unmapped = df['category_id'].isna().sum()
if unmapped:
    print(f"Warning: {unmapped} rows had category_ids not in the mapping and are now NaN.")

df.to_csv('f_annotated_categories_and_jobs.csv', index=False)

print("✅ Updated CSV written to 'f_annotated_categories_and_jobs.csv'")
