from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# 1. Load the dataset
dataset = load_dataset("ucirvine/sms_spam", split="train")

# 2. Convert dataset into lists
messages = dataset["sms"]
labels = dataset["label"]

# 3. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    messages,
    labels,
    test_size=0.2,
    random_state=42
)

# 4. Convert text into numbers
vectorizer = TfidfVectorizer()

X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# 5. Create the machine learning model
model = MultinomialNB()

# 6. Train the model
model.fit(X_train_vectors, y_train)

# 7. Test the model
predictions = model.predict(X_test_vectors)

accuracy = accuracy_score(y_test, predictions)

print("Model trained successfully!")
print("Accuracy:", accuracy)

# 8. Save the model
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model saved successfully!")