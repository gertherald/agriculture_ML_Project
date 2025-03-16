import sqlite3
import pandas as pd
import os

# Function to fetch evaluation metrics from the database
def fetch_evaluation_metrics(database_path):
    """Fetch evaluation metrics from the SQLite database."""
    connection = sqlite3.connect(database_path)
    query = "SELECT * FROM evaluation_metrics"
    df = pd.read_sql_query(query, connection)
    connection.close()
    return df

# Function to print the regressor with lowest MAE, lowest MSE, and highest Adjusted R-squared
def print_best_models(df):
    """Print the best regressor models based on MAE, MSE, and Adjusted R-squared."""
    
    # Finding the best model based on the lowest MAE
    best_mae_model = df.loc[df['mae'].idxmin()]
    print(f"Best Model by MAE: {best_mae_model['model_name']} with MAE = {best_mae_model['mae']:.4f}")
    
    # Finding the best model based on the lowest MSE
    best_mse_model = df.loc[df['mse_test'].idxmin()]
    print(f"Best Model by MSE: {best_mse_model['model_name']} with MSE = {best_mse_model['mse_test']:.4f}")
    
    # Finding the best model based on the highest Adjusted R-squared
    best_adjusted_r2_model = df.loc[df['adjusted_r2'].idxmax()]
    print(f"Best Model by Adjusted R-squared: {best_adjusted_r2_model['model_name']} with Adjusted R-squared = {best_adjusted_r2_model['adjusted_r2']:.4f}")

def run_evaluation_comparison():
    # Define the database path
    database_path = os.path.join(os.getcwd(), 'data', 'temp.db')
    
    # Fetch the evaluation metrics from the database
    evaluation_df = fetch_evaluation_metrics(database_path)
    
    # Display the evaluation metrics table
    print("\nEvaluation Metrics Table:")
    print(evaluation_df.to_string())
    
    # Print the best models based on MAE, MSE, and Adjusted R-squared
    print("\nBest Models Comparison:")
    print_best_models(evaluation_df)

if __name__ == '__main__':
    run_evaluation_comparison()