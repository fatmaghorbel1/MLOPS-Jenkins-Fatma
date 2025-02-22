import argparse
import mlflow
import mlflow.sklearn
import os
from pathlib import Path
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model

# Set the artifact path to a directory within the Jenkins workspace
mlflow.set_tracking_uri("file:///tmp/mlflow_artifacts")
mlflow.set_experiment("ML_Pipeline_Experiment")
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
                print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")
            else:
                print(f"Feature shape mismatch: Expected 19 features, but got {X_test.shape[1]} features.")
                return

if __name__ == "__main__":
    main()
