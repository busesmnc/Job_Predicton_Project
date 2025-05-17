import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
from scipy.sparse import hstack
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
print("Loading data...")
data_path = '/Users/busesomunncu/Desktop/Linkedln Job Prediction Project/final_annotated_categories_and_jobs.csv'
df = pd.read_csv(data_path)

# Define feature columns and target
feature_columns = [
    'technical_requirements',
    'soft_skills',
    'domain_knowledge',
    'education_requirement',
    'experience_level',
    'project_experience'
]
target = 'category_id'

# Initialize TF-IDF vectorizers for each text column
print("Preprocessing text data...")
vectorizers = {}
transformed_features = []

# Transform each text column using TF-IDF
for column in feature_columns:
    df[column] = df[column].fillna('')
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        token_pattern=r"(?u)\b\w+\b"
    )
    feature_matrix = vectorizer.fit_transform(df[column])
    transformed_features.append(feature_matrix)
    vectorizers[column] = vectorizer

# Combine all features into one sparse matrix
X = hstack(transformed_features)
y = df[target]

# Split the data
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create and train the logistic regression model
print("Training the model...")
model = LogisticRegression(
    multi_class='multinomial',
    max_iter=1000,
    random_state=42
)
model.fit(X_train, y_train)

# Make predictions
print("Making predictions...")
y_pred = model.predict(X_test)

# Calculate all evaluation metrics
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

# Create a more detailed report
print("\n3. Detailed Classification Report by Category:")
print("-" * 50)
unique_categories = sorted(df[target].unique())
for category in unique_categories:
    # Create binary classification for this category
    y_test_binary = (y_test == category)
    y_pred_binary = (y_pred == category)
    
    # Calculate metrics for this category
    precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_test_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)
    
    print(f"\nCategory {category}:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

# Print detailed classification report
print("\n4. Detailed Classification Report (All Categories):")
print("-" * 50)
print(classification_report(y_test, y_pred))

def predict_job_category(job_details):
    """
    Predict job category for new job details using Logistic Regression.
    
    Args:
        job_details (dict): Dictionary containing job details
        
    Returns:
        int: Predicted category ID
    """
    transformed_inputs = []
    
    for column in feature_columns:
        text = job_details.get(column, '')
        vectorizer = vectorizers[column]
        transformed = vectorizer.transform([text])
        transformed_inputs.append(transformed)
    
    X_new = hstack(transformed_inputs)
    
    # Get prediction
    prediction = model.predict(X_new)[0]
    # Get probability scores for all classes
    probabilities = model.predict_proba(X_new)[0]
    # Get the confidence score (probability) for the predicted class
    confidence = probabilities[model.classes_ == prediction][0]
    
    return int(prediction), confidence

# Example usage
print("\nTesting with example job posting:")
example_job = {
    'technical_requirements': 'python machine learning sql docker',
    'soft_skills': 'communication teamwork leadership',
    'domain_knowledge': 'data analysis statistics deep learning',
    'education_requirement': 'masters',
    'experience_level': 'senior',
    'project_experience': 'built machine learning models deployed to production'
}

predicted_category, confidence = predict_job_category(example_job)
print(f"Predicted category ID: {predicted_category}")
print(f"Confidence score: {confidence:.4f}")

# Save results to a file
with open('logistic_regression_results.txt', 'w') as f:
    f.write("Logistic Regression Model Evaluation Results\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("1. Classification Metrics:\n")
    f.write("-" * 30 + "\n")
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    f.write(f"Precision (weighted): {precision_score(y_test, y_pred, average='weighted'):.4f}\n")
    f.write(f"Recall/Sensitivity (weighted): {recall_score(y_test, y_pred, average='weighted'):.4f}\n")
    f.write(f"F1-measure (weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}\n\n")
    
    f.write("2. Confusion Matrix:\n")
    f.write("-" * 30 + "\n")
    f.write("Shape: " + str(conf_matrix.shape) + "\n")
    f.write("\n" + str(conf_matrix) + "\n\n")
    
    f.write("3. Detailed Classification Report:\n")
    f.write("-" * 30 + "\n")
    f.write(classification_report(y_test, y_pred)) 