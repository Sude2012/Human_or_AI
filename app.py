from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# MODELLER
tfidf = joblib.load("models/tfidf_vectorizer.pkl")
lr = joblib.load("models/logistic_model.pkl")
svm = joblib.load("models/svm_model.pkl")
rf = joblib.load("models/rf_model.pkl")

def predict_text(text):
    text_tfidf = tfidf.transform([text])

    lr_prob = lr.predict_proba(text_tfidf)[0][1]

    svm_score = svm.decision_function(text_tfidf)[0]
    svm_prob = 1 / (1 + np.exp(-svm_score))

    rf_prob = rf.predict_proba(text_tfidf)[0][1]

    return {
        "Logistic Regression": {
            "AI %": round(lr_prob * 100, 2),
            "Human %": round((1 - lr_prob) * 100, 2)
        },
        "SVM": {
            "AI %": round(svm_prob * 100, 2),
            "Human %": round((1 - svm_prob) * 100, 2)
        },
        "Random Forest": {
            "AI %": round(rf_prob * 100, 2),
            "Human %": round((1 - rf_prob) * 100, 2)
        }
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if "text" not in data:
        return jsonify({"error": "Text field missing"}), 400

    text = data["text"]

    if len(text) < 30:
        return jsonify({"error": "Text too short"}), 400

    return jsonify(predict_text(text))

if __name__ == "__main__":
    app.run(debug=True)
