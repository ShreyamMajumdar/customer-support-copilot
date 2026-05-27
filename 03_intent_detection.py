import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle
import os

df = pd.read_csv('data/tickets.csv')

vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
X = vectorizer.fit_transform(df['message'])
y = df['intent']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

print("Model trained!")
print("Accuracy:", round(accuracy * 100, 1), "%")

os.makedirs('data', exist_ok=True)
with open('data/model.pkl', 'wb') as f: pickle.dump(model,      f)
with open('data/vectorizer.pkl', 'wb') as f: pickle.dump(vectorizer, f)
print("Model saved.")

replies = {
    'Billing' : "Our billing team will contact you within 24 hours.",
    'Technical' : "Please clear cache and retry. Tech team will follow up.",
    'Shipping' : "Check your tracking number. Delivery takes 3-5 days.",
    'Account' : "Try resetting your password. We will help if issue continues.",
    'General' : "Visit our Help Center at support.example.com"
}

test_messages = [
    "I got charged twice this month",
    "App not opening on my phone",
    "Package has not come in 2 weeks",
    "I forgot my login password",
    "What is your return policy",
]

print("\nChatbot Test:")
print("-" * 50)
for msg in test_messages:
    vec = vectorizer.transform([msg])
    intent = model.predict(vec)[0]
    confidence = model.predict_proba(vec).max() * 100
    print("Message :", msg)
    print("Intent :", intent, " | Confidence:", round(confidence, 1), "%")
    print("Reply :", replies[intent])
    print("-" * 50)

plt.figure(figsize=(6, 5))
plt.bar(['Logistic Regression'], [accuracy * 100],
        color='#3498db', edgecolor='black', width=0.3)
plt.text(0, accuracy * 100 + 1,
         str(round(accuracy * 100, 1)) + "%",
         ha='center', fontsize=14)
plt.title('Intent Detection Accuracy', fontsize=14)
plt.ylabel('Accuracy (%)')
plt.ylim(0, 110)
plt.tight_layout()
plt.savefig('outputs/chart4_accuracy.png')
plt.show()
print("Chart saved.")