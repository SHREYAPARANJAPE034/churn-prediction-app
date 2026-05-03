import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from preprocess import load_and_preprocess

df, encoders = load_and_preprocess("data/churn.csv")

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier()
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# Save model + encoders
pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(encoders, open("models/encoders.pkl", "wb"))