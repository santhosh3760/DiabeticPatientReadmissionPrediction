import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, f1_score
import joblib

# Load data
def load_data(filepath):
    return pd.read_csv(filepath)

# Preprocess data
def preprocess_data(df):
    # Drop rows with missing target values
    df = df.dropna(subset=['readmitted'])

    # Separate features and target
    X = df.drop(columns=['readmitted'])
    y = df['readmitted'].astype(int)

    # Handling missing data by imputing with the median for numerical columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        X[col] = X[col].fillna(X[col].median())

    return X, y

# Evaluate the model
def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    return accuracy, precision, f1

# Main function to run the model training and saving
def main():
    # Load the data
    df = load_data('/Users/santhoshreddy/Desktop/vscode/DataMiningCourse/PatientReadmissionPredictionProject/preprocessed_data.csv')
    X, y = preprocess_data(df)

    # Split data into train+val and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Further split train+val into train and val sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    # Create a pipeline with scaling and logistic regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logistic', LogisticRegression())
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    accuracy, precision, f1 = evaluate_model(pipeline, X_val, y_val)
    print(f"Validation Accuracy: {accuracy}")
    print(f"Validation Precision: {precision}")
    print(f"Validation F1 Score: {f1}")

    # Save the model to disk
    joblib.dump(pipeline, 'logistic_regression_model.pkl')
    print("Model saved to logistic_regression_model.pkl")

if __name__ == "__main__":
    main()
