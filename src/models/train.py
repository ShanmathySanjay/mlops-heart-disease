import sys
import os

sys.path.append(os.path.abspath("."))

from matplotlib.pylab import rint
import mlflow
import mlflow.sklearn
import joblib

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from src.data.load_data import load_data
from src.features.preprocess import preprocess_data

if os.getenv("CI"):
    mlflow.set_tracking_uri("file:./mlruns")
else:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

def evaluate_model(name, y_test, preds):
    print(f"\n--- {name} ---")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds))
    print("Recall:", recall_score(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, preds))

def train():
    # Load + preprocess
    df = load_data()
    X, y = preprocess_data(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    # =========================
    # Logistic Regression
    # =========================
    with mlflow.start_run(run_name="Logistic Regression"):

        lr_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ])

        lr_pipeline.fit(X_train, y_train)
        lr_preds = lr_pipeline.predict(X_test)

        acc = accuracy_score(y_test, lr_preds)
        prec = precision_score(y_test, lr_preds)
        rec = recall_score(y_test, lr_preds)
        roc = roc_auc_score(y_test, lr_pipeline.predict_proba(X_test)[:, 1])
        cv_scores = cross_val_score(lr_pipeline, X, y, cv=5)

        print("\n--- Logistic Regression ---")
        print("Accuracy:", acc)
        print("Precision:", prec)
        print("Recall:", rec)
        print("ROC-AUC:", roc)
        print("Cross-val mean:", cv_scores.mean())
        print("Cross-val std:", cv_scores.std())

        # Log params
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("roc_auc", roc)
        mlflow.log_metric("cv_mean", cv_scores.mean())
        mlflow.log_metric("cv_std", cv_scores.std())

        # Confusion Matrix
        cm = confusion_matrix(y_test, lr_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title("Logistic Regression - Confusion Matrix")

        plt.savefig("lr_confusion_matrix.png")
        mlflow.log_artifact("lr_confusion_matrix.png")
        plt.close()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, lr_pipeline.predict_proba(X_test)[:, 1])
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Logistic Regression - ROC Curve")

        plt.savefig("lr_roc_curve.png")
        mlflow.log_artifact("lr_roc_curve.png")
        plt.close()

        # Log model
        # Save model manually
        joblib.dump(lr_pipeline, "lr_model.pkl")

        # Log as artifact
        mlflow.log_artifact("lr_model.pkl")

    # =========================
    # Random Forest
    # =========================
    with mlflow.start_run(run_name="Random Forest"):

        rf_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        rf_pipeline.fit(X_train, y_train)
        rf_preds = rf_pipeline.predict(X_test)

        acc = accuracy_score(y_test, rf_preds)
        prec = precision_score(y_test, rf_preds)
        rec = recall_score(y_test, rf_preds)
        roc = roc_auc_score(y_test, rf_pipeline.predict_proba(X_test)[:, 1])
        cv_scores = cross_val_score(rf_pipeline, X, y, cv=5)

        print("\n--- Random Forest ---")
        print("Accuracy:", acc)
        print("Precision:", prec)
        print("Recall:", rec)
        print("ROC-AUC:", roc)
        print("Cross-val mean:", cv_scores.mean())
        print("Cross-val std:", cv_scores.std())

        # Log params
        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", 100)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("roc_auc", roc)
        mlflow.log_metric("cv_mean", cv_scores.mean())
        mlflow.log_metric("cv_std", cv_scores.std())

        # Confusion Matrix
        cm = confusion_matrix(y_test, rf_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title("Random Forest - Confusion Matrix")

        plt.savefig("rf_confusion_matrix.png")
        mlflow.log_artifact("rf_confusion_matrix.png")
        plt.close()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, rf_pipeline.predict_proba(X_test)[:, 1])
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Random Forest - ROC Curve")

        plt.savefig("rf_roc_curve.png")
        mlflow.log_artifact("rf_roc_curve.png")
        plt.close()

        # Log model

        # Save model manually
        joblib.dump(rf_pipeline, "rf_model.pkl")

        # Log as artifact
        mlflow.log_artifact("rf_model.pkl")

        # =========================
        # Model Selection
        # =========================

        print("\nSelecting best model based on ROC-AUC...")

        # Compare models
        models = {
            "LogisticRegression": (lr_pipeline, roc_auc_score(y_test, lr_preds)),
            "RandomForest": (rf_pipeline, roc_auc_score(y_test, rf_preds))
        }

        best_model_name = max(models, key=lambda x: models[x][1])
        best_model, best_score = models[best_model_name]

        print(f"Best model: {best_model_name} with ROC-AUC: {best_score}")

        # Save final model
        joblib.dump(best_model, "model.pkl")
        print("Best model saved as model.pkl")

        # Log best model to MLflow
        mlflow.set_experiment("Best Model Selection")
        with mlflow.start_run(run_name="Best Model", nested=True):
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metric("best_roc_auc", best_score)
            mlflow.log_artifact("model.pkl")

if __name__ == "__main__":
    train()