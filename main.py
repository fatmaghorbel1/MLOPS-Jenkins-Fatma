import argparse
import mlflow
import mlflow.sklearn
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from pathlib import Path
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model

artifact_path = os.path.expanduser("~/mlflow_artifacts")
os.makedirs(artifact_path, exist_ok=True)  # Ensure the directory exists
mlflow.set_tracking_uri("file:///home/fatma/mlflow_artifacts")
mlflow.set_experiment("Churn_Prediction")
def plot_roc_curve(y_true, y_pred_proba, file_name="roc_curve.png"):
    """
    Plot ROC curve and save it as an image.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.savefig(file_name)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, file_name="confusion_matrix.png"):
    """
    Plot confusion matrix and save it as an image.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Churn", "Churn"], yticklabels=["Not Churn", "Churn"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(file_name)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="ML Pipeline Execution")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--save", action="store_true", help="Save the trained model")
    parser.add_argument("--load", action="store_true", help="Load the saved model")
    args = parser.parse_args()

    # File paths for training and testing data
    train_file = "churn-bigml-80.csv"
    test_file = "churn-bigml-20.csv"

    # Prepare the data
    X_train, X_test, y_train, y_test, label_encoders = prepare_data(train_file, test_file)

    # Debugging: Print shapes of training and test data
    print(f"Training data shape: {X_train.shape}")  # Ensure this prints (n_samples, 19)
    print(f"Test data shape: {X_test.shape}")  # Ensure this prints (n_samples, 19)

    # Start an MLflow experiment
    mlflow.set_experiment("Churn_Prediction")

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 6)

        if args.train:
            # Train the model using the XGBoost model
            model = train_model(X_train, y_train)

            # Save the trained model to disk if save argument is passed
            if args.save:
                save_model(model, "model.joblib")
                print("Model saved successfully.")

                # Log the model as an MLflow artifact
                mlflow.sklearn.log_model(model, "model")
                print("Model logged to MLflow.")

        if args.evaluate:
            # Load the trained model from disk
            if args.load:
                model = load_model()
                print("Model loaded successfully.")
            else:
                print("Please use --load to load a pre-trained model.")
                return

            # Debugging: Ensure X_test has the correct number of features
            if X_test.shape[1] == 19:
                # Evaluate the model using the test data
                accuracy, precision, recall = evaluate_model(model, X_test, y_test)

                # Log metrics
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)

		# Generate predictions and probabilities for ROC curve
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]  

 		# Plot and log ROC curve
                plot_roc_curve(y_test, y_pred_proba, "roc_curve.png")
                mlflow.log_artifact("roc_curve.png")
                print("ROC curve logged as artifact")

                # Plot and log confusion matrix
                plot_confusion_matrix(y_test, y_pred, "confusion_matrix.png")
                mlflow.log_artifact("confusion_matrix.png")
                print("Confusion matrix logged as artifact")

                # Log artifacts (e.g., feature importance plot)
                import matplotlib.pyplot as plt
                importances = model.feature_importances_
                plt.barh(X_test.columns, importances)
                plt.savefig("feature_importance.png")
                print("Plot saved as feature_importance.png")
                mlflow.log_artifact("feature_importance.png")
                print("Plot logged as artifact")
                print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")
            else:
                print(f"Feature shape mismatch: Expected 19 features, but got {X_test.shape[1]} features.")
                return

if __name__ == "__main__":
    main()
