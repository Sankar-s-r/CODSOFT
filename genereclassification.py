import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ==========================================
# 1. Load the Data
# ==========================================
train_path = 'Genre Classification Dataset\\train_data.txt' 
train_df = pd.read_csv(train_path, sep=' ::: ', engine='python', 
                       names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'])

# ==========================================
# 2. LEVEL 2: Group Similar Genres
# ==========================================
# We map confusing, overlapping genres into a broader category
genre_mapping = {
    'biography': 'drama_history',
    'history': 'drama_history',
    'drama': 'drama_history'
}
# If a genre is in the mapping, replace it; otherwise, keep the original
train_df['MAPPED_GENRE'] = train_df['GENRE'].replace(genre_mapping)

print("New Genre Distribution (Top 5):")
print(train_df['MAPPED_GENRE'].value_counts().head(5))

# ==========================================
# 3. Clean the Data
# ==========================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

train_df['CLEAN_DESC'] = train_df['DESCRIPTION'].apply(clean_text)

X = train_df['CLEAN_DESC']
y = train_df['MAPPED_GENRE'] # Use the new grouped labels
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 4. LEVEL 1 & 2: Upgraded Pipeline
# ==========================================
# Add custom movie-related stop words to the standard English list
custom_stop_words = list(ENGLISH_STOP_WORDS) + ['movie', 'film', 'story', 'life', 'world', 'man', 'woman']

# Build the new pipeline with N-grams and LinearSVC
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=20000,           # Increased feature limit to accommodate 2-word phrases
        stop_words=custom_stop_words, # Using our expanded stop words list
        ngram_range=(1, 2)            # Look at single words AND 2-word pairs
    )),
    ('clf', LinearSVC(class_weight='balanced', dual='auto', max_iter=2000)) # Upgraded classifier
])

print("\nTraining the upgraded model...")
model_pipeline.fit(X_train, y_train)

# ==========================================
# 5. Internal Validation
# ==========================================
print("\nEvaluating the upgraded model...")
val_predictions = model_pipeline.predict(X_val)

print("Accuracy:", accuracy_score(y_val, val_predictions))
print("\nClassification Report:\n", classification_report(y_val, val_predictions))

# ==========================================
# 6. Final Blind Test (Inference)
# ==========================================
print("\n--- Starting Final Blind Test ---")

# 1. Load the Test Data 
# Make sure 'test_data.txt' matches the exact name of your test file
test_path = 'Genre Classification Dataset\\test_data.txt' 
test_df = pd.read_csv(test_path, sep=' ::: ', engine='python', 
                      names=['ID', 'TITLE', 'DESCRIPTION'])

# 2. Clean the test descriptions using the EXACT SAME function from training
print("Cleaning test descriptions...")
test_df['CLEAN_DESC'] = test_df['DESCRIPTION'].apply(clean_text)

# 3. Predict the missing genres using our upgraded pipeline
# The pipeline automatically handles the TF-IDF vectorization for the new text!
print("Predicting genres for the Test Data...")
test_df['PREDICTED_GENRE'] = model_pipeline.predict(test_df['CLEAN_DESC'])

# 4. Save the final results to a CSV file
submission_df = test_df[['ID', 'TITLE', 'PREDICTED_GENRE']]
submission_df.to_csv('final_test_predictions.csv', index=False)

print("\nSuccess! Predictions saved to 'final_test_predictions.csv'")