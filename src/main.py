import subprocess
import os

def run_script(script_name):
    """Function to run a Python script."""
    script_path = os.path.join(os.getcwd(), script_name)
    try:
        result = subprocess.run(['python3', script_path], check=True, capture_output=True, text=True)
        print(f"Successfully ran {script_name}")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}")
        print(e.stderr)

if __name__ == '__main__':
    # Run Data Ingestion and Preprocessing Script
    print("Running Data Ingestion and Preprocessing...")
    run_script('Data_Ingestion_And_Preprocessing.py')
    print("Preprocessing Cleaned Data into PCA for Plant Type-Stage Prediction")
    run_script('Plant_PCA.py')

    
    # Run Regressors (Gradient Boosting Regressor, Random Forest Regressor, Linear Regression)
    print("Running Regressors...")
    print("Training and Testing Gradient Boosting Regressor...")
    run_script('Gradient_Boosting_Regressor.py')
    print("Training and Testing Random Forest Regressor...")
    run_script('Random_Forest_Regressor.py')
    print("Training and Testing Linear Regressor...")
    run_script('Linear_Regression.py')

    # Run Regression Evaluation Script
    print("Running Regression Evaluation...")
    run_script('Regression_Evaluation.py')

    # Run Classifiers with Label Encoding
    print("Running Classifiers (Label)...")
    print("Compare Raw cleaned data and PCA data")
    print("Training and Testing Raw Decision Tree Classifier (Label)...")
    run_script('Decision_Tree_Classifier_Label_Raw.py')
    print("Training and Testing PCA Decision Tree Classifier (Label)...")
    run_script('Decision_Tree_Classifier_Label_PCA.py')
    print("Accuracy of PCA model is higher than Raw Cleaned Data model using the same model. Thus, we will use PCA data for the next models")
    print("Training and Testing K Nearest Neighbours Classifier (Label)...")
    run_script('K_Nearest_Neighbours_Classifier_Label_PCA.py')
    print("Training and Testing Random Forest Classifier (Label)...")
    run_script('Random_Forest_Classifier_Label_PCA.py')

    # Run Classifiers with OneHot Encoding
    print("Running Classifiers (OneHot)...")
    print("Training and Testing Decision Tree Classifier (OneHot)...")
    run_script('Decision_Tree_Classifier_OneHot_PCA.py')
    print("Training and Testing K Nearest Neighbours Classifier (OneHot)...")
    run_script('K_Nearest_Neighbours_Classifier_OneHot_PCA.py')


    # Run Classifier Evaluation Script
    print("Running Classifier Evaluation...")
    run_script('Classifier_Evaluation.py')

    # Run Application Script
    print("Running Application Tester")
    run_script('application_test.py')