import pandas as pd
import re
import json
import numpy as np

pd.set_option('display.max_columns', None)

df_labeled = pd.read_csv("/Users/busesomunncu/Desktop/Linkedln Job Prediction Project/final_labeled-data.csv")
df_labeled_from_excel = pd.read_csv("/Users/busesomunncu/Desktop/Linkedln Job Prediction Project/labeled_job_data_updated.csv")
df_categories = pd.read_csv("/Users/busesomunncu/Desktop/Linkedln Job Prediction Project/cats_id.csv")
df_main = pd.read_csv("/Users/busesomunncu/Desktop/Linkedln Job Prediction Project/text_mining_data_updated.csv")
df_ =pd.read_csv("/Users/busesomunncu/Desktop/Linkedln Job Prediction Project/labeled_job_data_updated.csv")



df = pd.read_csv("deneme.csv")
for i in df["predicted_tech_requirements"]:
    print("hi")
    if i == None:
        pass
    else:
        print(df["predicted_tech_requirements"])

# CSV dosyası
# data = pd.read_csv('jobs_tolabel.csv')
""" 
# Excel'e
# data.to_excel('jobs_tolabel.xlsx', index=False)


df = pd.read_excel('/Users/busesomunncu/Desktop/Linkedln Job Prediction Project/labeled_jobs_112.xlsx')

# Kategorileri birleştirerek yeni sütunlar ekleyelim
def extract_categories(categories_str):
    try:
        categories_dict = json.loads(categories_str)
        return categories_dict
    except json.JSONDecodeError:
        return {}

# categories sütununu işleyip her başlık için yeni sütunlar oluşturuyoruz
categories_expanded = df['categories'].apply(extract_categories)

# Her kategori başlığına karşılık yeni sütunlar oluşturuyoruz
for category in categories_expanded[0].keys():
    df[category] = categories_expanded.apply(lambda x: x.get(category, None))

# Artık 'categories' sütunu gereksiz, onu silebiliriz
df.drop(columns=['categories'], inplace=True)

# CSV'ye kaydetme
df.to_csv("/Users/busesomunncu/Desktop/Linkedln Job Prediction Project/labeled_job_data.csv", index=False)

df['experience_level'] = df['experience_level'].replace('Entry-level', 'Junior')
df.replace(['none', 'None'], np.nan, inplace=True) 

df.columns = df.columns.str.lower()

def remove_emojis(text):
    return re.sub(r'[^\w\s,]', '', text)  # Bu regex sadece harf, rakam, boşluk ve virgül dışındaki karakterleri kaldırır

df = df.applymap(lambda x: remove_emojis(str(x).lower()) if isinstance(x, str) else x)

df.to_csv("/Users/busesomunncu/Desktop/Linkedln Job Prediction Project/text_mining_data_updated.csv", index=False)

# 2. Kategorilerin bulunduğu CSV dosyasını yükleyin
df_categories = pd.read_csv("/Users/busesomunncu/Desktop/Linkedln Job Prediction Project/categories.csv")
df_categories.columns = df_categories.columns.str.lower()

def remove_emojis(text):
    return re.sub(r'[^\w\s,]', '', text)  # Bu regex sadece harf, rakam, boşluk ve virgül dışındaki karakterleri kaldırır

df_categories = df_categories.applymap(lambda x: remove_emojis(str(x).lower()) if isinstance(x, str) else x)
df_main = df_main.merge(df_categories[['job_title', 'category_name']], how='left', on='job_title')

# 4. Yeni dosyayı kaydedin
df_main.to_csv("/Users/busesomunncu/Desktop/Linkedln Job Prediction Project/final_labeled-data.csv", index=False)

# Tamamıyla aynı olan satırları bul
duplicates = df[df.duplicated()]
duplicates2 = df[df.duplicated(subset=["job_title", "category_name"])]
print(duplicates)
print(duplicates2)

print(df2.info())  # Sonuçları kontrol edebilirsiniz

df_categories.columns = df_categories.columns.str.lower()

def remove_emojis(text):
    return re.sub(r'[^\w\s,]', '', text)  # Bu regex sadece harf, rakam, boşluk ve virgül dışındaki karakterleri kaldırır

df_categories = df_categories.applymap(lambda x: remove_emojis(str(x).lower()) if isinstance(x, str) else x)

# Sıralama sütunu ekle
df['index'] = df.index
df_categories['index'] = df_categories.index

# Şimdi merge işlemini yapalım
df = df.merge(df_categories[['index', 'category_name']], how='left', on='index')

# 'index' sütununu kaldır
df = df.drop(columns=['index'])

# Merge sonrası satır sayısını kontrol et
print("Yeni satır sayısı:", len(df))
df.to_csv("/Users/busesomunncu/Desktop/Linkedln Job Prediction Project/final_labeled-data.csv", index=False)

df_categories.columns = df_categories.columns.str.lower()

def remove_emojis(text):
    return re.sub(r'[^\w\s,]', '', text)  # Bu regex sadece harf, rakam, boşluk ve virgül dışındaki karakterleri kaldırır

df_categories = df_categories.applymap(lambda x: remove_emojis(str(x).lower()) if isinstance(x, str) else x)

# Sıralama sütunu ekle
df['index'] = df.index
df_categories['index'] = df_categories.index

# Şimdi merge işlemini yapalım
df = df.merge(df_categories[['index', 'category_id']], how='left', on='index')

# 'index' sütununu kaldır
df = df.drop(columns=['index'])

# Merge sonrası satır sayısını kontrol et
print("Yeni satır sayısı:", len(df))
df.to_csv("/Users/busesomunncu/Desktop/Linkedln Job Prediction Project/final_labeled-data2.csv", index=False)
"""

