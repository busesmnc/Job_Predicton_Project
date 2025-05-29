import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    classification_report
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

print("Loading data...")
data_path = '/Users/busesomunncu/Desktop/Linkedln Job Prediction Project/f_annotated_categories_and_jobs.csv'
df = pd.read_csv(data_path)

feature_columns = [
    'technical_requirements',
    'soft_skills',
    'domain_knowledge',
    'education_requirement',
    'experience_level',
    'project_experience'
]
target = 'category_id'

print("Preprocessing text data...")
vectorizers = {}
transformed_features = []

for column in feature_columns:
    df[column] = df[column].fillna('')
    vectorizer = TfidfVectorizer(
        max_features=1000,  # Reduced from 2000
        stop_words='english',
        token_pattern=r"(?u)\b\w+\b",
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 1)  # Only unigrams
    )
    feature_matrix = vectorizer.fit_transform(df[column])
    transformed_features.append(feature_matrix)
    vectorizers[column] = vectorizer

X = hstack(transformed_features)
y = df[target]

print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Calculate class weights
print("Computing class weights...")
classes = np.array(sorted(y.unique()))
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)
class_weight_dict = dict(zip(classes, class_weights))

# SMOTE
print("Applying SMOTE for class balancing...")
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

base_model = LogisticRegression(random_state=42)

print("Performing hyperparameter tuning...")
param_grid = {
    'C': [0.01, 0.05, 0.1, 0.5, 1],  # Smaller C values for stronger regularization
    'solver': ['lbfgs', 'saga'],
    'max_iter': [2000, 3000],
    'class_weight': [None, 'balanced']
}

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train_res, y_train_res)

print("\nBest parameters found:")
print(grid_search.best_params_)
print(f"\nBest cross-validation score: {grid_search.best_score_:.4f}")

model = grid_search.best_estimator_

print("\nPerforming cross-validation with best model...")
cv_scores = cross_val_score(model, X_train_res, y_train_res, cv=5, scoring='f1_weighted')
print(f"Cross-validation F1 scores: {cv_scores}")
print(f"Average CV F1 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

model.fit(X_train_res, y_train_res)

# Save
print("\nSaving model and test data...")
joblib.dump(model, 'logistic_regression_model.joblib')
joblib.dump(X_test, 'X_test.joblib')
joblib.dump(y_test, 'y_test.joblib')

print("Making predictions...")
y_pred = model.predict(X_test)

print("\nEvaluation Metrics:")
print("=" * 50)

# Classification Metrics
print("\n1. Classification Metrics:")
print("-" * 30)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision (weighted): {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall/Sensitivity (weighted): {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1-measure (weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}")

# Confusion Matrix
print("\n2. Confusion Matrix:")
print("-" * 30)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix Shape:", conf_matrix.shape)
print("\nConfusion Matrix:")
print(conf_matrix)

# report
print("\n3. Detailed Classification Report by Category:")
print("-" * 50)
print(classification_report(y_test, y_pred))

def predict_job_category(job_details):
    """
    Predict job category for new job details using improved Logistic Regression.
    
    Args:
        job_details (dict): Dictionary containing job details
        
    Returns:
        tuple: (predicted_category, confidence, top_3_predictions)
    """
    transformed_inputs = []
    
    for column in feature_columns:
        text = job_details.get(column, '')
        vectorizer = vectorizers[column]
        transformed = vectorizer.transform([text])
        transformed_inputs.append(transformed)
    
    X_new = hstack(transformed_inputs)
    
    prediction = model.predict(X_new)[0]
    
    probabilities = model.predict_proba(X_new)[0]
    
    confidence = probabilities[model.classes_ == prediction][0]
    
    # 3 predictions
    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    top_3_predictions = [
        (int(model.classes_[idx]), float(probabilities[idx]))
        for idx in top_3_indices
    ]
    
    return int(prediction), confidence, top_3_predictions

# Example usage
print("\nTesting with example job posting:")
example_job = {
    'technical_requirements': 'research methodology statistical analysis python r machine learning',
    'soft_skills': 'communication critical thinking teamwork presentation skills',
    'domain_knowledge': 'curriculum development higher education pedagogy academic writing ethics',
    'education_requirement': 'junior',
    'experience_level': 'mid level',  # Örnek: 3-5 years academia experience
    'project_experience': 'grant proposal writing interdisciplinary research project publication review process'
}

predicted_category, confidence, top_3 = predict_job_category(example_job)
print(f"\nPredicted category ID: {predicted_category}")
print(f"Confidence score: {confidence:.4f}")
print("\nTop 3 predictions:")
for category, prob in top_3:
    print(f"Category {category}: {prob:.4f}")

# Save results to a file
with open('improved_logistic_regression_results.txt', 'w') as f:
    f.write("Improved Logistic Regression Model Evaluation Results\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("Cross-validation Results:\n")
    f.write("-" * 30 + "\n")
    f.write(f"Average CV F1 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n\n")
    
    f.write("Classification Metrics:\n")
    f.write("-" * 30 + "\n")
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    f.write(f"Precision (weighted): {precision_score(y_test, y_pred, average='weighted'):.4f}\n")
    f.write(f"Recall/Sensitivity (weighted): {recall_score(y_test, y_pred, average='weighted'):.4f}\n")
    f.write(f"F1-measure (weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}\n\n")
    
    f.write("Confusion Matrix:\n")
    f.write("-" * 30 + "\n")
    f.write("Shape: " + str(conf_matrix.shape) + "\n")
    f.write("\n" + str(conf_matrix) + "\n\n")
    
    f.write("Detailed Classification Report:\n")
    f.write("-" * 30 + "\n")
    f.write(classification_report(y_test, y_pred)) 