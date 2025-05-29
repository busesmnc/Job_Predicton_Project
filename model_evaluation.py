import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    roc_curve,
    auc
)
from sklearn.model_selection import learning_curve
import joblib

print("Loading model and data...")
model = joblib.load('logistic_regression_model.joblib')
X_test = joblib.load('X_test.joblib')
y_test = joblib.load('y_test.joblib')

# Test Set Performance
print("\n1. Test Set Performance Analysis")
print("=" * 50)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Calculate metrics
accuracy = np.mean(y_pred == y_test)
print(f"\nOverall Accuracy: {accuracy:.4f}")

# Detailed classification report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix Analysis
print("\n2. Confusion Matrix Analysis")
print("=" * 50)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')
plt.close()

# Learning Curve Analysis
print("\n3. Learning Curve Analysis")
print("=" * 50)
train_sizes, train_scores, test_scores = learning_curve(
    model, X_test, y_test, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='f1_weighted'
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score')
plt.plot(train_sizes, test_mean, label='Cross-validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
plt.title('Learning Curve')
plt.xlabel('Training Examples')
plt.ylabel('Score')
plt.legend(loc='best')
plt.grid(True)
plt.savefig('learning_curve.png')
plt.close()

# Category-wise Analysis
print("\n4. Category-wise Analysis")
print("=" * 50)
categories = np.unique(y_test)
for category in categories:
    category_mask = y_test == category
    category_pred = y_pred[category_mask]
    category_true = y_test[category_mask]
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        category_true, category_pred, average='weighted'
    )
    
    print(f"\nCategory {category}:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# Feature Importance Analysis
print("\n5. Feature Importance Analysis")
print("=" * 50)
if hasattr(model, 'coef_'):
    feature_importance = np.abs(model.coef_)
    for i, category in enumerate(categories):
        top_features_idx = np.argsort(feature_importance[i])[-10:]  # Top 10 features
        print(f"\nTop 10 important features for Category {category}:")
        for idx in top_features_idx:
            print(f"Feature {idx}: {feature_importance[i][idx]:.4f}")

# Save results to a file
with open('model_evaluation_results.txt', 'w') as f:
    f.write("Model Evaluation Results\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("1. Test Set Performance\n")
    f.write("-" * 30 + "\n")
    f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))
    
    f.write("\n2. Category-wise Analysis\n")
    f.write("-" * 30 + "\n")
    for category in categories:
        category_mask = y_test == category
        category_pred = y_pred[category_mask]
        category_true = y_test[category_mask]
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            category_true, category_pred, average='weighted'
        )
        
        f.write(f"\nCategory {category}:\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

print("\nAnalysis complete! Results have been saved to 'model_evaluation_results.txt'")
print("Visualizations have been saved as 'confusion_matrix.png' and 'learning_curve.png'") 