import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# ✅ Paths
data_path = "data"
model_path = "models"

# Load datasets
fake_df = pd.read_csv(os.path.join(data_path, "Fake.csv"))
true_df = pd.read_csv(os.path.join(data_path, "True.csv"))

# Add labels
fake_df["label"] = 0   # Fake
true_df["label"] = 1   # Real

# Combine
df = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)

# Use only the 'text' column for now
X = df["text"]
y = df["label"]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model + vectorizer
if not os.path.exists(model_path):
    os.makedirs(model_path)

with open(os.path.join(model_path, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(model_path, "vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved!")
