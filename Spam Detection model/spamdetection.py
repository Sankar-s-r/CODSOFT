import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix
import json

# Load data - often this specific dataset uses latin-1 encoding
df = pd.read_csv('spam detection dataset\\spam.csv', encoding='latin-1')

# Step 3: Explore & Step 4: Prepare Data
# Drop unnamed columns that are full of NaNs
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

exploration_info = {
    'shape': df.shape,
    'spam_count': int((df['label'] == 'spam').sum()),
    'ham_count': int((df['label'] == 'ham').sum())
}

# Encode labels
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label_num'], test_size=0.2, random_state=42, stratify=df['label_num'])

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Step 5 & 6: Explore Models
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Support Vector Machine': SVC(probability=True, random_state=42)
}

results = {}
plt.figure(figsize=(10, 6))

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    y_prob = model.predict_proba(X_test_tfidf)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Store results safely
    results[name] = {'Accuracy': float(acc), 'AUC': float(roc_auc)}
    
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Spam Detection Models')
plt.legend(loc="lower right")
plt.savefig('spam_roc_curve.png')
plt.close()

# Let's get the top spam words using Logistic Regression coefficients
log_reg = models['Logistic Regression']
feature_names = tfidf.get_feature_names_out()
coefs = log_reg.coef_[0]

top_spam_idx = np.argsort(coefs)[-10:]
top_spam_words = [feature_names[i] for i in top_spam_idx]

print(json.dumps({'exploration': exploration_info, 'results': results, 'top_spam_words': top_spam_words}, indent=4))