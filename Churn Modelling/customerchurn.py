import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import json

# Load data
df = pd.read_csv('Churn Modelling Dataset//Churn_Modelling.csv')

# Step 3: Exploration info
exploration_info = {
    'shape': df.shape,
    'columns': df.columns.tolist(),
    'churn_rate': float(df['Exited'].mean())
}

# Step 4: Prepare Data
df_clean = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
df_encoded = pd.get_dummies(df_clean, columns=['Geography', 'Gender'], drop_first=True)

X = df_encoded.drop('Exited', axis=1)
y = df_encoded['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5 & 6: Explore Models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

results = {}
plt.figure(figsize=(10, 6))

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    results[name] = {'Accuracy': acc, 'AUC': roc_auc}
    
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Churn Prediction Models')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.close()

# Feature importance for Gradient Boosting (usually slightly better or similar to RF)
gb_model = models['Gradient Boosting']
importances = gb_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(10, 6))
plt.title('Feature Importances (Gradient Boosting)')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45, ha='right')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print(json.dumps({'exploration': exploration_info, 'results': results},indent=4))