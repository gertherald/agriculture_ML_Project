import sqlite3
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import OneHotEncoder
import os

# Function to fetch cleaned data from the database
def fetch_cleaned_data(database_path, table_name):
    """Fetch cleaned data from the SQLite database."""
    connection = sqlite3.connect(database_path)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, connection)
    connection.close()
    return df

# Function to store evaluation metrics into the database for classifiers
def store_classifier_evaluation_metrics(database_path, model_name, accuracy):
    """Store the evaluation metrics for the classifier in the database."""
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()
    
    # Create a table for storing evaluation metrics if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS evaluation_metric_2 (
        model_name TEXT,
        accuracy REAL
    )
    ''')

    # Insert the evaluation metrics into the table
    cursor.execute('''
    INSERT INTO evaluation_metric_2 (model_name, accuracy)
    VALUES (?, ?)
    ''', (model_name, accuracy))
    
    connection.commit()
    connection.close()

def run_classifier_evaluation():
    # Define the database path and table names
    database_path = os.path.join(os.getcwd(), 'data', 'temp.db')
    table_name = 'plant_pca_df' 
    
    # Fetch the cleaned data
    cleaned_df = fetch_cleaned_data(database_path, table_name)
    
    # Define predictors and target
    # Define the predictors and target
    target = 'Plant Type-Stage'

    # Prepare feature matrix X and target vector y
    X = cleaned_df.drop(columns = [target])
    y = cleaned_df[target]

    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)

    # Fit and transform the target variable (Plant Type-Stage)
    y_reshaped = y.values.reshape(-1, 1)
    y_encoded = encoder.fit_transform(y_reshaped)

    # Split the data into train (80%) and test (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Create a pipeline with the Decision Tree model
    model_pipeline = Pipeline(steps=[
        ('classifier', MultiOutputClassifier(DecisionTreeClassifier(random_state=42)))  # Using DecisionTreeClassifier
    ])

    # Define the hyperparameters to tune for Decision Tree
    param_grid = {
        'classifier__estimator__max_depth': [3, 5, 7, None],
        'classifier__estimator__min_samples_split': [2, 5, 10],
        'classifier__estimator__min_samples_leaf': [1, 2, 4],
        'classifier__estimator__criterion': ['gini', 'entropy']
    }

    # Set up GridSearchCV to search over the hyperparameters and evaluate using accuracy
    grid_search = GridSearchCV(estimator=model_pipeline, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Perform GridSearchCV on the training set (80% of the data)
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters found by GridSearchCV
    print(f"Best Hyperparameters: {grid_search.best_params_}")

    # Get the best model found by GridSearchCV
    best_model = grid_search.best_estimator_

    # Predict on the test data (20% of the data)
    y_pred = best_model.predict(X_test)

    # Calculate the accuracy score on the test data
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on Test Data: {accuracy:.4f}")

    # Store the evaluation metrics in the database for the classifier (append to evaluation_metric_2)
    store_classifier_evaluation_metrics(database_path, 'DecisionTreeClassifierOneHot', accuracy)

if __name__ == '__main__':
    run_classifier_evaluation()