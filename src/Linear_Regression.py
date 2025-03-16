import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Define a function to fetch cleaned data from the database
def fetch_cleaned_data(database_path, table_name):
    """Fetch cleaned data from the SQLite database."""
    connection = sqlite3.connect(database_path)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, connection)
    connection.close()
    return df

# Define a function to store evaluation metrics into the database
def store_evaluation_metrics(database_path, model_name, mse_test, mse_baseline, r2, adjusted_r2, mae):
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()
    
    # Create a table for storing evaluation metrics if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS evaluation_metrics (
        model_name TEXT,
        mse_test REAL,
        mse_baseline REAL,
        r2 REAL,
        adjusted_r2 REAL,
        mae REAL
    )
    ''')
    
    # Insert the evaluation metrics into the table
    cursor.execute('''
    INSERT INTO evaluation_metrics (model_name, mse_test, mse_baseline, r2, adjusted_r2, mae)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (model_name, mse_test, mse_baseline, r2, adjusted_r2, mae))
    
    connection.commit()
    connection.close()

def run_evaluation():
    # Define the database path and table names
    database_path = os.path.join(os.getcwd(), 'data', 'temp.db')
    table_name = 'cleaned_data'  # Assuming this is the table storing the cleaned data
    
    # Fetch the cleaned data
    cleaned_df = fetch_cleaned_data(database_path, table_name)
    
    # Define predictors and target
    predictors = ['Light Intensity Sensor (lux)', 'Humidity Sensor (%)', 'Nutrient K Sensor (ppm)', 'CO2 Sensor (ppm)', 'Plant Type', 'Labelled Plant Stage']
    target = 'Temperature Sensor (°C)'

    # Prepare feature matrix X and target vector y
    X = cleaned_df[predictors]
    y = cleaned_df[target]

    # Split the data into train (80%) and test (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

    # Define categorical feature (for OneHotEncoding)
    cat_onehot = ['Plant Type']

    # Create a column transformer to handle categorical features with OneHotEncoding and scaling for numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), cat_onehot),  # OneHotEncoding for categorical feature
            ('num', StandardScaler(), [col for col in predictors if col not in cat_onehot])  # Scaling numerical features
        ]
    )

    # Create a pipeline with preprocessing and the regressor model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Define the hyperparameters to tune (though LinearRegression has fewer hyperparameters)
    param_grid = {
        'regressor__fit_intercept': [True, False],
    }

    # Set up GridSearchCV to search over the hyperparameters and evaluate using 5-fold cross-validation
    grid_search = GridSearchCV(estimator=model_pipeline, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

    # Perform GridSearchCV on the training set (80% of the data)
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters found by GridSearchCV
    print(f"Best Hyperparameters: {grid_search.best_params_}")

    # Get the best model found by GridSearchCV
    best_model = grid_search.best_estimator_

    # Predict on the test data (20% of the data)
    y_pred = best_model.predict(X_test)

    # Calculate the Mean Squared Error on the test data
    mse_test = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on Test Data: {mse_test}")

    # Calculate the baseline MSE using the mean of the target variable
    y_mean_pred = [y.mean()] * len(y_test)  # Predict the mean value for all test data points
    mse_baseline = mean_squared_error(y_test, y_mean_pred)
    print(f"Baseline Mean Squared Error: {mse_baseline}")

    # Calculate R-squared value
    r2 = r2_score(y_test, y_pred)
    print(f"R-squared value: {r2:.4f}")

    # Calculate Adjusted R-squared value
    n = len(y_test)  # Number of data points
    p = X_test.shape[1]  # Number of predictors (features)

    # Adjusted R-squared formula
    adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
    print(f"Adjusted R-squared value: {adjusted_r2:.4f}")

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae:.4f}")

    # Store the evaluation metrics in the database
    store_evaluation_metrics(database_path, 'LinearRegression', mse_test, mse_baseline, r2, adjusted_r2, mae)

if __name__ == '__main__':
    run_evaluation()