import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

def train_and_evaluate():
    print("[1/5] Loading AG News dataset...")
    dataset = load_dataset("fancyzhx/ag_news")
    X_train = dataset["train"]["text"]
    y_train = dataset["train"]["label"]
    X_test = dataset["test"]["text"]
    y_test = dataset["test"]["label"]

    print("[2/5] Building TF-IDF + Logistic Regression pipeline...")
    # Production-ready pipeline: handles vectorization + model in a single artifact
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=25000, ngram_range=(1, 2), stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1))
    ])

    print("[3/5] Training model (this will take ~30-60 seconds)...")
    pipeline.fit(X_train, y_train)

    print("[4/5] Evaluating on test set (7,600 samples)...")
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=LABEL_NAMES, digits=4)
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT:")
    print("="*50)
    print(report)

    # Save metrics report to results/
    os.makedirs("results", exist_ok=True)
    with open("results/sklearn_report.txt", "w") as f:
        f.write(report)

    # Save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix — TF-IDF + Logistic Regression")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=200)
    plt.close()
    print("Saved confusion matrix to results/confusion_matrix.png")

    # Save the trained model artifact
    os.makedirs("models", exist_ok=True)
    model_path = "models/tfidf_logreg.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Saved model artifact to {model_path}")

    # [5/5] Learning Curve (Diagnostic for Bias vs Variance)
    print("\n[5/5] Generating learning curve on subset (diagnosing bias vs variance)...")
    # Using a 20k subset to compute learning curve fast on CPU
    subset_indices = np.random.choice(len(X_train), size=20000, replace=False)
    X_sub = [X_train[i] for i in subset_indices]
    y_sub = [y_train[i] for i in subset_indices]

    train_sizes, train_scores, val_scores = learning_curve(
        pipeline,
        X_sub,
        y_sub,
        cv=3,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring="accuracy",
        n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, "o-", color="r", label="Training score")
    plt.plot(train_sizes, val_mean, "o-", color="g", label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve (TF-IDF + Logistic Regression)")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/learning_curve.png", dpi=200)
    plt.close()
    print("Saved learning curve to results/learning_curve.png")

if __name__ == "__main__":
    train_and_evaluate()
