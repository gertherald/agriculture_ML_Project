import sqlite3
import pandas as pd
import os

# Function to fetch evaluation metrics from the database
def fetch_evaluation_metrics(database_path):
    """Fetch the evaluation metrics table from the SQLite database."""
    connection = sqlite3.connect(database_path)
    query = "SELECT * FROM evaluation_metric_2"
    df = pd.read_sql_query(query, connection)
    connection.close()
    return df

# Function to print the method with the highest accuracy score
def print_highest_accuracy_method(database_path):
    """Print the method with the highest accuracy score from the evaluation metrics."""
    df = fetch_evaluation_metrics(database_path)
    
    if not df.empty:
        # Find the model with the highest accuracy
        highest_accuracy_row = df.loc[df['accuracy'].idxmax()]
        highest_accuracy_model = highest_accuracy_row['model_name']
        highest_accuracy = highest_accuracy_row['accuracy']
        print(f"Model with Highest Accuracy: {highest_accuracy_model}")
        print(f"Accuracy: {highest_accuracy:.4f}")
    else:
        print("No evaluation metrics available.")

# Function to display the evaluation metrics table
def display_evaluation_table(database_path):
    """Display the evaluation metrics table."""
    df = fetch_evaluation_metrics(database_path)
    
    if not df.empty:
        print("Evaluation Metrics Table:")
        print(df)
    else:
        print("No evaluation metrics available.")

if __name__ == '__main__':
    database_path = os.path.join(os.getcwd(), 'data', 'temp.db')
    
    # Display the evaluation metrics table
    display_evaluation_table(database_path)
    
    # Print the method with the highest accuracy score
    print_highest_accuracy_method(database_path)