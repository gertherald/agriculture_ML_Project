import sqlite3
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import os

def load_data_from_db(database_path, table_name):
    """Fetch data from the specified table in the SQLite database."""
    connection = sqlite3.connect(database_path)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, connection)
    connection.close()
    return df

def clean_data(df):
    """Preprocess and clean the data."""
    
    # Clean columns by ensuring they are numeric and removing non-numeric values (e.g., 'ppm')
    df['Nutrient N Sensor (ppm)'] = pd.to_numeric(df['Nutrient N Sensor (ppm)'].astype(str).str.split(' ').str[0], errors='coerce')
    df['Nutrient P Sensor (ppm)'] = pd.to_numeric(df['Nutrient P Sensor (ppm)'].astype(str).str.split(' ').str[0], errors='coerce')
    df['Nutrient K Sensor (ppm)'] = pd.to_numeric(df['Nutrient K Sensor (ppm)'].astype(str).str.split(' ').str[0], errors='coerce')

    # Select only numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    # Remove rows where any of the numeric columns contain negative values
    df = df[~(df[numeric_columns] < 0).any(axis=1)]

    # Drop NaN values in all columns except for 'Humidity Sensor (%)'
    df = df.dropna(subset=[col for col in df.columns if col != 'Humidity Sensor (%)'])

    # Impute missing values for 'Humidity Sensor (%)' using KNNImputer
    imputer = KNNImputer(n_neighbors=5)
    df['Humidity Sensor (%)'] = imputer.fit_transform(df[['Humidity Sensor (%)']])

    # Change all unique Plant Type to Pascal Case/ Title (Start of each word is Capitalized)
    df['Plant Type'] = df['Plant Type'].str.title()

    # Change all unique Plant Stage to Pascal Case/ Title (Start of each word is Capitalized)
    df['Plant Stage'] = df['Plant Stage'].str.title()

    # Combine 'Plant Type' and 'Plant Stage' into one categorical column
    df['Plant Type-Stage'] = df['Plant Type'] + '-' + df['Plant Stage']

    # Start to Label Encode Plant Stages
    mapping = {'Seedling': 1, 'Vegetative': 2, 'Maturity': 3}
    df['Labelled Plant Stage'] = df['Plant Stage'].map(mapping)

    return df

def save_data_to_db(df, database_path, table_name):
    """Save the cleaned data to the SQLite database in a specified table."""
    connection = sqlite3.connect(database_path)
    df.to_sql(table_name, connection, if_exists='replace', index=False)
    connection.close()

def main():
    # Absolute path to the SQLite database (change this based on your environment)
    database_fetch = os.path.join(os.getcwd(), 'data', 'calls.db')
    database_return = os.path.join(os.getcwd(), 'data', 'temp.db')
    # Fetch the first table name from the database (or choose another table name)
    connection = sqlite3.connect(database_fetch)
    cursor = connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    connection.close()

    # Print the table names
    for table in tables:
        print(f"Table name: {table[0]}")  # Prints each table name

    # Choose the first table or any specific table
    first_table_name = tables[0][0]  # For example, use the first table
    print(f"Fetching data from table: {first_table_name}")

    # Load the data from the SQLite database
    df = load_data_from_db(database_fetch, first_table_name)

    # Clean the data
    cleaned_df = clean_data(df)

    # Save the cleaned data back to the SQLite database
    save_data_to_db(cleaned_df, database_return, 'cleaned_data')

    print("Data preprocessing and saving completed successfully.")

if __name__ == '__main__':
    main()