import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

# =========================
# 1️⃣ VERİ YÜKLEME (ÖNCE!)
# =========================
ai_df = pd.read_csv("data/ai2_data.csv")
human_df = pd.read_csv("data/human_data kopyası.csv")

print("\n=== AI SAMPLE ===")
print(ai_df["text"].iloc[0][:300])

print("\n=== HUMAN SAMPLE ===")
print(human_df["text"].iloc[0][:300])

# ⚠️ BURAYI DİKKATLİ OKU
# Eğer yukarıda:
# - AI SAMPLE insan gibi
# - HUMAN SAMPLE robotik gibi görünüyorsa
# ETİKETLER TERS DEMEKTİR

# =========================
# 2️⃣ LABEL ATAMA
# =========================
ai_df["label"] = 0
human_df["label"] = 1


df = pd.concat([ai_df, human_df], ignore_index=True)

print("\nLabel dağılımı:")
print(df["label"].value_counts())

# =========================
# 3️⃣ VERİ TEMİZLEME
# =========================
df.dropna(subset=["text"], inplace=True)
df["text"] = df["text"].astype(str)
df = df[df["text"].str.len() > 30]
df.drop_duplicates(subset=["text"], inplace=True)

print("Temizlenmiş veri boyutu:", df.shape)

# =========================
# 4️⃣ TRAIN / TEST SPLIT
# =========================
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 5️⃣ TF-IDF
# =========================
tfidf = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

# =========================
# 6️⃣ LOGISTIC REGRESSION
# =========================
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_tfidf, y_train)

y_pred_lr = lr.predict(X_test_tfidf)
y_prob_lr = lr.predict_proba(X_test_tfidf)[:, 1]

print("\nLOGISTIC REGRESSION")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

joblib.dump(lr, "models/logistic_model.pkl")

# =========================
# 7️⃣ SVM
# =========================
svm = LinearSVC()
svm.fit(X_train_tfidf, y_train)

y_pred_svm = svm.predict(X_test_tfidf)

print("\nSVM")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

joblib.dump(svm, "models/svm_model.pkl")

# =========================
# 8️⃣ RANDOM FOREST
# =========================
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_tfidf, y_train)

y_pred_rf = rf.predict(X_test_tfidf)
y_prob_rf = rf.predict_proba(X_test_tfidf)[:, 1]

print("\nRANDOM FOREST")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

joblib.dump(rf, "models/rf_model.pkl")

# =========================
# 9️⃣ CONFUSION MATRIX
# =========================
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_lr,
    display_labels=["Human", "AI"]
)
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# =========================
# 🔟 ROC CURVE
# =========================
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure()
plt.plot(fpr_lr, tpr_lr, label=f"LR (AUC={roc_auc_lr:.2f})")
plt.plot(fpr_rf, tpr_rf, label=f"RF (AUC={roc_auc_rf:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.legend()
plt.title("ROC Curve")
plt.show()

print("\n✅ MODELLER EĞİTİLDİ")
