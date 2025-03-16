import sqlite3
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import joblib

def load_data_from_db(database_path, table_name):
    """Fetch data from the specified table in the SQLite database."""
    connection = sqlite3.connect(database_path)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, connection)
    connection.close()
    return df

def save_data_to_db(df, database_path, table_name):
    """Save the PCA transformed data to the SQLite database in a specified table."""
    connection = sqlite3.connect(database_path)
    df.to_sql(table_name, connection, if_exists='replace', index=False)
    connection.close()

def main():
    # Absolute path to the SQLite database (adjust the path as needed)
    database_path = os.path.join(os.getcwd(), 'data', 'temp.db')

    # Load the cleaned_data table
    cleaned_df = load_data_from_db(database_path, 'cleaned_data')

    # Define the target variable
    target = 'Plant Type-Stage'

    # Select all columns except target and 'Plant Stage' for X
    X = cleaned_df.drop(columns=[target, 'Plant Stage', 'Plant Type', 'System Location Code', 'Previous Cycle Plant Type', 'Labelled Plant Stage'])

    # Separate the target variable from the features (use all columns except target and 'Plant Stage')
    y = cleaned_df[target]

    print(X)

    # Standardize the feature data (important for PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit PCA and calculate the explained variance for all components
    pca = PCA()
    pca.fit(X_scaled)

    # Cumulative explained variance ratio
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

    # Find the number of components required to explain at least 95% of the variance
    threshold_variance = 0.95
    optimal_n_components = np.argmax(cumulative_explained_variance >= threshold_variance) + 1  # Add 1 because index starts from 0

    print(f"The optimal number of components to explain at least {threshold_variance*100}% variance is: {optimal_n_components}")

    # Fit PCA using the optimal number of components
    pca = PCA(n_components=optimal_n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Create a DataFrame with the PCA components and the target variable
    plant_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(optimal_n_components)])
    plant_pca_df[target] = y

    # After PCA, reset the index of y to ensure alignment
    plant_pca_df[target] = y.reset_index(drop=True)

    # Print the explained variance ratio for each component
    print("\nExplained Variance Ratio per Component:")
    for i in range(optimal_n_components):
        print(f"PC{i+1}: {pca.explained_variance_ratio_[i]:.4f}")

    # Calculate the cumulative explained variance ratio
    print(f"\nCumulative Explained Variance by the top {optimal_n_components} components: {cumulative_explained_variance[optimal_n_components-1]:.4f}")

    # Print the loadings (coefficients) that show how each feature affects each principal component
    print("\nPrincipal Component Loadings (Feature Influence on Each Component):")
    loadings = pd.DataFrame(pca.components_, columns=X.columns, index=[f'PC{i+1}' for i in range(optimal_n_components)])
    print(loadings)

    # Save the PCA transformed data to the database as a new table
    save_data_to_db(plant_pca_df, database_path, 'plant_pca_df')

    # Display the new DataFrame with PCA components and target
    print(plant_pca_df.head())

    joblib.dump(scaler, 'scaler.pkl')  # Save the scaler
    joblib.dump(pca, 'pca_model.pkl')  # Save the PCA model

if __name__ == '__main__':
    main()