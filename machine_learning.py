
# =============================================================================
# Tez Projem: Metin Özellik Sütunlarının Vektörleştirilmesi,
# Dengesiz Veri Çözümleri ve Model Değerlendirme
# =============================================================================

# 0. Gerekli Paket Kurulumu (Konsolda Çalıştırın)
# pip install imbalanced-learn

# 1. Gerekli Kütüphaneler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
# Show all columns
pd.set_option('display.max_columns', None)
# 2. Veri Yükleme
# Güncel CSV dosya yolunu belirtin
data_path = '/Users/busesomunncu/Desktop/Linkedln Job Prediction Project/final_annotated_categories_and_jobs.csv'
df = pd.read_csv(data_path)

# 3. Özellik ve Hedef Değişkenain Belirlenmesi
y = df['category_id']
feature_cols = [col for col in df.columns if col not in ['id', 'category_id']]

# 4. Metin Özelliklerinin Vektörleştirilmesi
vectorizers = {}
sparse_list = []
for col in feature_cols:
    vect = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    mat = vect.fit_transform(df[col].fillna(''))
    vectorizers[col] = vect
    sparse_list.append(mat)
# Tüm sütun matrislerini yatayda birleştir
X_sparse = hstack(sparse_list)

# 5. Eğitim/Test Verisine Bölme (%80 / %20)
X_train, X_test, y_train, y_test = train_test_split(
    X_sparse, y,
    train_size=0.8,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Dengesiz Veri Çözümü: SMOTE ile Oversampling
# Az örnekli sınıflar için k_neighbors=3 ile ayarlandı
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 7. Dengesiz Veri Çözümü: Class Weight Hesaplama
classes = np.array(sorted(y.unique()))
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y.values
)
class_weight_dict = dict(zip(classes, class_weights))

# 8. Model Kurulumu ve 5 Katlı CV (F1 Weighted)
model = LogisticRegression(
    solver='liblinear',
    class_weight=class_weight_dict,
    max_iter=1000,
    random_state=42
)
cv_scores = cross_val_score(
    model,
    X_train_res,
    y_train_res,
    cv=5,
    scoring='f1_weighted'
)
print('5 Katlı CV F1 Skorları:', cv_scores)
print('Ortalama CV F1 Skor:', cv_scores.mean())

# 9. Model Eğitimi ve Test Seti Değerlendirme
model.fit(X_train_res, y_train_res)
y_pred = model.predict(X_test)

print('Accuracy :', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred, average='weighted'))
print('Recall   :', recall_score(y_test, y_pred, average='weighted'))
print('F1 Score :', f1_score(y_test, y_pred, average='weighted'))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

# 10. Tek Örnek Tahmin Fonksiyonu

def predict_category(new_sample: dict) -> int:
    """
    new_sample: {'column_name': 'keyword1 keyword2', ...}
    return: tahmin edilen category_id
    """
    sparse_cols = []
    for col in feature_cols:
        vect = vectorizers[col]
        text = new_sample.get(col, '')
        sparse_cols.append(vect.transform([text]))
    X_new = hstack(sparse_cols)
    return model.predict(X_new)[0]

# Örnek kullanım:
# sample = {col: 'keyword1 keyword2' for col in feature_cols}
# print(predict_category(sample))

# 1. Örnek veri satırı
sample = {
    'technical_requirements': 'kubernetes docker spark tensorflow',
    'soft_skills': 'adaptability analytical thinking time management',
    'domain_knowledge': 'network security embedded systems ux ui design',
    'education_requirement': 'bachelor',
    'experience_level': 'junior',
    'project_experience': 'chatbot machine learning model'
}

# 2. Model ile tahmin
predicted_category = predict_category(sample)
print(f"Tahmin edilen category_id: {predicted_category}")

# ID - İsim eşleştirmesi
category_map = {
    1: 'Software Engineer',
    2: 'Computer Engineer',
    3: 'Data Analyst / Data Scientist',
    4: 'Web / Mobile Developer',
    5: 'IT Support Specialist',
    6: 'DevOps Engineer',
    7: 'Cybersecurity Specialist',
    8: 'Cloud Engineer',
    9: 'Network Engineer / Systems Administrator',
    10: 'Database Administrator (DBA)',
    11: 'Embedded Systems / Hardware Engineer',
    12: 'Site Reliability Engineer (SRE)',
    13: 'Robotics Engineer',
    14: 'IoT Engineer',
    15: 'Blockchain Developer',
    16: 'AR/VR Developer',
    17: 'AI / Machine Learning Engineer',
    18: 'Product Manager',
    19: 'Business Analyst',
    20: 'System Analyst',
    21: 'Project Manager',
    22: 'Business Development Specialist',
    23: 'Financial Analyst',
    24: 'Operations Manager',
    25: 'Sales and Marketing Manager',
    26: 'Customer Relations Specialist',
    27: 'Production/Manufacturing Engineer',
    28: 'Game Designer',
    29: 'Logistics / Supply Chain Manager',
    30: 'University Professor / Academic Staff'
}


# Kategori ismini yazdırma
category_name = category_map.get(predicted_category, 'Unknown')
print(f"Kategori ismi: {category_name}")

sample2 = {
    'technical_requirements': 'linux sql kubernetes node.js',
    'soft_skills': 'problem solving communication time management',
    'domain_knowledge': 'embedded systems software testing ux ui design',
    'education_requirement': 'high school',
    'experience_level': 'junior',
    'project_experience': 'inventory management system cloud migration project'
}

# 2. Model ile Tahmin ve Sonuç Yazdırma
pred_id2 = predict_category(sample2)
cat_name2 = category_map.get(pred_id2, 'Unknown')
print(f"Tahmin edilen category_id: {pred_id2}")
print(f"Kategori ismi: {cat_name2}")

sample3 = {
    'technical_requirements': 'docker java c++ spark',
    'soft_skills': 'analytical thinking problem solving creativity',
    'domain_knowledge': 'software testing embedded systems ai modeling',
    'education_requirement': 'bachelor',
    'experience_level': 'junior',
    'project_experience': 'cloud security audit report e-commerce platform'
}

# 2. Tahmin ve Sonuç
pred_id3 = predict_category(sample3)
cat_name3 = category_map.get(pred_id3, 'Unknown')
print(f"Tahmin edilen category_id: {pred_id3}")
print(f"Kategori ismi: {cat_name3}")

sample4 = {
    'technical_requirements': 'react html docker sql',
    'soft_skills': 'communication problem solving teamwork',
    'domain_knowledge': 'software testing database optimization ai modeling',
    'education_requirement': 'phd',
    'experience_level': 'junior',
    'project_experience': 'inventory management system security audit report'
}

# 2. Tahmin ve Sonuç
pred_id4 = predict_category(sample4)
cat_name4 = category_map.get(pred_id4, 'Unknown')
print(f"Tahmin edilen category_id: {pred_id4}")
print(f"Kategori ismi: {cat_name4}")


sample5 = {
    'technical_requirements': 'research methodology statistical analysis python r machine learning',
    'soft_skills': 'communication critical thinking teamwork presentation skills',
    'domain_knowledge': 'curriculum development higher education pedagogy academic writing ethics',
    'education_requirement': 'phd',
    'experience_level': 'mid level',  # Örnek: 3-5 years academia experience
    'project_experience': 'grant proposal writing interdisciplinary research project publication review process'
}

# 2. Tahmin ve Sonuç
pred_id5 = predict_category(sample5)
cat_name5 = category_map.get(pred_id5, 'Unknown')
print(f"Tahmin edilen category_id: {pred_id5}")
print(f"Kategori ismi: {cat_name5}")


sample_cloud_engineer = {
    'technical_requirements': 'aws azure docker kubernetes terraform ci/cd',
    'soft_skills': 'problem solving collaboration communication adaptability',
    'domain_knowledge': 'cloud architecture microservices distributed systems security compliance',
    'education_requirement': 'bachelor',
    'experience_level': 'mid level',  # Örnek: 3-5 years cloud experience
    'project_experience': 'infrastructure automation high availability design cost optimization'
}

# Tahmin ve çıktı
pred_id_cloud = predict_category(sample_cloud_engineer)
cat_name_cloud = category_map.get(pred_id_cloud, 'Unknown')
print(f"Tahmin edilen category_id: {pred_id_cloud}")
print(f"Kategori ismi: {cat_name_cloud}")
