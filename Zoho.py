# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample dataset of movie reviews
# Assuming you have a dataset with columns: 'review' (text) and 'label' (0 for negative, 1 for positive)
data = {
    'review': [
        "This movie was fantastic! I loved it!",
        "Worst movie I've ever seen. So boring.",
        "Absolutely wonderful, would watch again!",
        "Horrible, not worth the time.",
        "It was a great experience, the actors were amazing.",
        "Terrible plot, I didn't like it at all."
    ],
    'label': [1, 0, 1, 0, 1, 0]
}

# Create DataFrame
df = pd.DataFrame(data)

# Step 1: Preprocess the data (convert text to feature vectors)
X = df['review']
y = df['label']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Vectorize the text (convert text to numerical format using TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 2: Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Step 3: Make predictions
y_pred = model.predict(X_test_tfidf)

# Step 4: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Display confusion matrix and classification report
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
