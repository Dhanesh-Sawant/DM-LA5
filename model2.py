import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Create synthetic dataset for demonstration
np.random.seed(42)
n_samples = 1000

# Generate synthetic data
data = {
    'CGPA': np.random.uniform(6.0, 10.0, n_samples),
    'GRE_Score': np.random.randint(260, 340, n_samples),
    'TOEFL_Score': np.random.randint(80, 120, n_samples),
    'Research_Papers': np.random.randint(0, 5, n_samples),
    'Mini_Projects': np.random.randint(1, 6, n_samples),
    'Internships': np.random.randint(0, 4, n_samples)
}

# Create DataFrame
df = pd.DataFrame(data)

# Generate target variable (Admission Status) based on weighted features
weights = {
    'CGPA': 0.3,
    'GRE_Score': 0.25,
    'TOEFL_Score': 0.2,
    'Research_Papers': 0.1,
    'Mini_Projects': 0.1,
    'Internships': 0.05
}

# Calculate weighted sum and create binary target
weighted_sum = sum(df[feature] * weight for feature, weight in weights.items())
threshold = np.percentile(weighted_sum, 70)  # 30% acceptance rate
df['Admitted'] = (weighted_sum > threshold).astype(int)

# Split features and target
X = df.drop('Admitted', axis=1)
y = df['Admitted']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

# Calculate feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': abs(model.coef_[0])
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# Create visualizations
plt.figure(figsize=(15, 10))

# 1. Feature Importance Plot
plt.subplot(2, 2, 1)
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance in Admission Prediction')

# 2. Confusion Matrix
plt.subplot(2, 2, 2)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# 3. Distribution of Admission Probability
plt.subplot(2, 2, 3)
sns.histplot(y_pred_prob, bins=30)
plt.title('Distribution of Admission Probabilities')
plt.xlabel('Probability of Admission')
plt.ylabel('Count')

# 4. CGPA vs GRE Score with Admission Status
plt.subplot(2, 2, 4)
admitted = X_test[y_test == 1]
not_admitted = X_test[y_test == 0]
plt.scatter(admitted['CGPA'], admitted['GRE_Score'], c='green', label='Admitted', alpha=0.5)
plt.scatter(not_admitted['CGPA'], not_admitted['GRE_Score'], c='red', label='Not Admitted', alpha=0.5)
plt.xlabel('CGPA')
plt.ylabel('GRE Score')
plt.title('CGPA vs GRE Score by Admission Status')
plt.legend()

plt.tight_layout()

# Print model evaluation metrics
print("\nModel Evaluation Metrics:")
print("-------------------------")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Function to predict admission probability for a new student
def predict_admission(student_data):
    """
    Predict admission probability for a new student
    
    Parameters:
    student_data: dict with keys for each feature
    
    Returns:
    probability of admission
    """
    # Convert student data to DataFrame
    student_df = pd.DataFrame([student_data])
    
    # Scale the features
    student_scaled = scaler.transform(student_df)
    
    # Get probability prediction
    prob = model.predict_proba(student_scaled)[0][1]
    
    return prob

# Example usage of prediction function
example_student = {
    'CGPA': 9.2,
    'GRE_Score': 325,
    'TOEFL_Score': 110,
    'Research_Papers': 2,
    'Mini_Projects': 4,
    'Internships': 2
}

admission_prob = predict_admission(example_student)
print("\nExample Prediction:")
print("------------------")
print(f"Probability of admission for example student: {admission_prob:.2%}")

# Print feature coefficients
print("\nFeature Coefficients:")
print("--------------------")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature}: {coef:.4f}")