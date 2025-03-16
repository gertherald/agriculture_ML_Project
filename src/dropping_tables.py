import sqlite3
import os

def drop_table(database_path, table_name):
    """Drops the specified table from the SQLite database."""
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    # Drop the table if it exists
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

    # Commit the changes and close the connection
    connection.commit()
    connection.close()
    print(f"Table '{table_name}' has been dropped.")

# Example usage
database_path = os.path.join(os.getcwd(), 'data', 'temp.db')
drop_table(database_path, 'evaluation_metric_2')
drop_table(database_path, 'evaluation_metrics')
drop_table(database_path, 'cleaned_data')
drop_table(database_path, 'plant_pca_df')