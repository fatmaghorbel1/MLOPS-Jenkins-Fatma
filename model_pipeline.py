import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score
import xgboost as xgb  # Import XGBoost library

# Load the dataset
df = pd.read_csv("churn-bigml-80.csv")  
print(df.head())  # Print first 5 rows
print(df.info())  # Check column data types

def prepare_data(train_file, test_file):
    """Load and preprocess the dataset"""
    train_df = pd.read_csv(train_file)  # Load training data
    test_df = pd.read_csv(test_file)    # Load testing data

    # Identify categorical columns
    categorical_cols = train_df.select_dtypes(include=['object']).columns

    # Convert categorical columns to numeric
    label_encoders = {}  # Store label encoders for later use
    for col in categorical_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])  # Train the encoder and transform the train data

        # Handle unseen values in the test set
        test_df[col] = test_df[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

        label_encoders[col] = le  # Save the encoder for future use (if you want to encode test data similarly)

    # Split features and labels
    X_train = train_df.drop("Churn", axis=1)  # Remove target column for feature set
    y_train = train_df["Churn"]  # Store target column for training

    X_test = test_df.drop("Churn", axis=1)  # Remove target column for test feature set
    y_test = test_df["Churn"]  # Store target column for testing

    return X_train, X_test, y_train, y_test, label_encoders

def train_model(X_train, y_train):
    """Train a machine learning model using XGBoost."""
    model = xgb.XGBClassifier(
        n_estimators=100,  # Number of boosting rounds (trees)
        max_depth=6,        # Maximum depth of each tree (controls model complexity)
        learning_rate=0.1,  # Step size shrinking to prevent overfitting
        random_state=42,    # For reproducibility
        use_label_encoder=False  # Disable the label encoder (to avoid warnings)
    )
    model.fit(X_train, y_train)  # Train the model on the training data
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's accuracy."""
    y_pred = model.predict(X_test)  # Predict on the test data
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return accuracy, precision, recall
    

def save_model(model, filename="model.pkl"):
    """Save the trained model."""
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

def load_model(filename="model.pkl"):
    """Load a saved model."""
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model
