import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack

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
    # Fill NaN values with empty string
    df[column] = df[column].fillna('')
    
    # Create and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=1000,  # Limit features to prevent overfitting
        stop_words='english',
        token_pattern=r"(?u)\b\w+\b"  # Include single-character words
    )
    
    # Transform the text column
    feature_matrix = vectorizer.fit_transform(df[column])
    transformed_features.append(feature_matrix)
    vectorizers[column] = vectorizer

# Combine all features into one sparse matrix
X = hstack(transformed_features)
y = df[target]

# Split the data into training and testing sets
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

# Calculate and print metrics
print("\nModel Performance Metrics:")
print("-" * 30)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision (weighted): {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall (weighted): {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1 Score (weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Function to predict category for new job descriptions
def predict_job_category(job_details):
    """
    Predict job category for new job details.
    
    Args:
        job_details (dict): Dictionary containing job details with the same columns as training data
        
    Returns:
        int: Predicted category ID
    """
    # Transform each feature using saved vectorizers
    transformed_inputs = []
    
    for column in feature_columns:
        text = job_details.get(column, '')
        vectorizer = vectorizers[column]
        transformed = vectorizer.transform([text])
        transformed_inputs.append(transformed)
    
    # Combine all transformed features
    X_new = hstack(transformed_inputs)
    
    # Make prediction
    return model.predict(X_new)[0]

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
print(f"Predicted category ID: {predicted_category}") 