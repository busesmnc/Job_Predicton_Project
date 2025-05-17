import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error, 
    r2_score,
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score
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
    random_state=42
)

# Create and train the linear regression model
print("Training the model...")
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
print("Making predictions...")
y_pred = model.predict(X_test)
y_pred_rounded = np.round(y_pred).astype(int)

# Calculate all evaluation metrics
print("\nEvaluation Metrics:")
print("=" * 50)

# 1. Regression Metrics
print("\n1. Regression Metrics:")
print("-" * 30)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R-squared Score: {r2_score(y_test, y_pred):.4f}")

# 2. Classification Metrics (after rounding)
print("\n2. Classification Metrics:")
print("-" * 30)
print(f"Accuracy: {accuracy_score(y_test, y_pred_rounded):.4f}")
print(f"Precision (weighted): {precision_score(y_test, y_pred_rounded, average='weighted'):.4f}")
print(f"Recall/Sensitivity (weighted): {recall_score(y_test, y_pred_rounded, average='weighted'):.4f}")
print(f"F1-measure (weighted): {f1_score(y_test, y_pred_rounded, average='weighted'):.4f}")

# 3. Confusion Matrix
print("\n3. Confusion Matrix:")
print("-" * 30)
conf_matrix = confusion_matrix(y_test, y_pred_rounded)
print("Confusion Matrix Shape:", conf_matrix.shape)
print("\nConfusion Matrix:")
print(conf_matrix)

# Create a more detailed report
print("\n4. Detailed Classification Report by Category:")
print("-" * 50)
unique_categories = sorted(df[target].unique())
for category in unique_categories:
    # Create binary classification for this category
    y_test_binary = (y_test == category)
    y_pred_binary = (y_pred_rounded == category)
    
    # Calculate metrics for this category
    precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_test_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)
    
    print(f"\nCategory {category}:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

def predict_job_category(job_details):
    """
    Predict job category for new job details using Linear Regression.
    
    Args:
        job_details (dict): Dictionary containing job details
        
    Returns:
        int: Predicted category ID (rounded to nearest integer)
    """
    transformed_inputs = []
    
    for column in feature_columns:
        text = job_details.get(column, '')
        vectorizer = vectorizers[column]
        transformed = vectorizer.transform([text])
        transformed_inputs.append(transformed)
    
    X_new = hstack(transformed_inputs)
    
    # Get prediction and round to nearest integer
    prediction = model.predict(X_new)[0]
    return int(round(prediction))

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

predicted_category = predict_job_category(example_job)
raw_prediction = model.predict(hstack([v.transform([example_job[col]]) for col, v in vectorizers.items()]))[0]
print(f"Raw predicted value: {raw_prediction:.2f}")
print(f"Rounded prediction (category ID): {predicted_category}")

# Save results to a file
with open('evaluation_results.txt', 'w') as f:
    f.write("Linear Regression Model Evaluation Results\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("1. Regression Metrics:\n")
    f.write("-" * 30 + "\n")
    f.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}\n")
    f.write(f"R-squared Score: {r2_score(y_test, y_pred):.4f}\n\n")
    
    f.write("2. Classification Metrics:\n")
    f.write("-" * 30 + "\n")
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred_rounded):.4f}\n")
    f.write(f"Precision (weighted): {precision_score(y_test, y_pred_rounded, average='weighted'):.4f}\n")
    f.write(f"Recall/Sensitivity (weighted): {recall_score(y_test, y_pred_rounded, average='weighted'):.4f}\n")
    f.write(f"F1-measure (weighted): {f1_score(y_test, y_pred_rounded, average='weighted'):.4f}\n\n")
    
    f.write("3. Confusion Matrix:\n")
    f.write("-" * 30 + "\n")
    f.write("Shape: " + str(conf_matrix.shape) + "\n")
    f.write("\n" + str(conf_matrix) + "\n") 