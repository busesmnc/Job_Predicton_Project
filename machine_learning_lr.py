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
category_name = category_map.get(predicted_category, 'Unknown')
print(f"Kategori ismi: {category_name}")

sample = {
    'technical_requirements': 'kubernetes docker spark tensorflow',
    'soft_skills': 'adaptability analytical thinking time management',
    'domain_knowledge': 'network security embedded systems ux ui design',
    'education_requirement': 'bachelor',
    'experience_level': 'junior',
    'project_experience': 'chatbot machine learning model'
}

# 2. Model ile tahmin
predicted_category = predict_job_category(sample)
print(f"Predicted category ID: {predicted_category}")
category_name = category_map.get(predicted_category, 'Unknown')
print(f"Kategori ismi: {category_name}")

sample3 = {
    'technical_requirements': 'docker java c++ spark',
    'soft_skills': 'analytical thinking problem solving creativity',
    'domain_knowledge': 'software testing embedded systems ai modeling',
    'education_requirement': 'bachelor',
    'experience_level': 'junior',
    'project_experience': 'cloud security audit report e-commerce platform'
}

# 2. Tahmin ve Sonuç
predicted_category = predict_job_category(sample3)
print(f"Predicted category ID: {predicted_category}")
category_name = category_map.get(predicted_category, 'Unknown')
print(f"Kategori ismi: {category_name}")


