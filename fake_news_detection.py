import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("news.csv")
df = df.dropna()
df["text"] = df["text"].astype(str)

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_df=0.7
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

while True:
    text = input("\nEnter news text (type exit to stop): ")
    if text.lower() == "exit":
        break

    vec = vectorizer.transform([text])
    probs = model.predict_proba(vec)[0]

    real_prob = probs[list(model.classes_).index("REAL")]
    fake_prob = probs[list(model.classes_).index("FAKE")]

    print(f"Confidence -> REAL: {real_prob:.2f}, FAKE: {fake_prob:.2f}")

    if real_prob > fake_prob:
        print("Prediction: REAL")
    else:
        print("Prediction: FAKE")
