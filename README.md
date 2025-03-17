# Machine Learning Pipeline for Plant Data

## a. Full Name and Email Address
Fullname: Gerald Chan Weiheng  
Email: geraldchanwh@gmail.com

## b. Overview of the Submitted Folder and Folder Structure
This project implements a machine learning pipeline to process and analyze plant-related sensor data. The folder contains various scripts that handle different parts of the pipeline, including data ingestion and preprocessing, machine learning models, and model evaluations. The folder structure is as follows:
## Folder Structure
.github  
src/  
│  
├── data/  
│   ├── calls.db  
│   └── temp.db  
├── application_test.py  
├── Classifier_Evaluation.py  
├── Data_Ingestion_And_Preprocessing.py  
├── Decision_Tree_Classifer_Label_PCA.py  
├── Decision_Tree_Classifer_Label_Raw.py  
├── Decision_Tree_Classifer_OneHot_PCA.py  
├── dropping_tables.py  
├── Gradient_Boosting_Regressor.py  
├── K_Nearest_Neighbours_Classifier_Label_PCA.py  
├── K_Nearest_Neighbours_Classifier_OneHot_PCA.py  
├── knn_classifier_model_trained.pkl  
├── Linear_Regression.py  
├── main.py  
├── pca_model.pkl  
├── Plant_PCA.py  
├── Random_Forest_Classifier_Label_PCA.py  
├── Random_Forest_Regressor.py  
├── random_forest_temperature_regressor_model_trained.pkl  
├── Regression_Evaluation.py  
└── scaler.pkl  
eda.ipynb  
README.md  
requirements.txt  
run.sh  

### Explanation:

- The **`src/`** folder contains all the main Python scripts required to run the machine learning pipeline.
- The **`data/`** folder can store any input data files to be fetched.
- **`eda.ipynb`** is the Jupyter notebook for exploratory data analysis.
- **`README.md`** is the file you're reading to describe the project and its structure.
- **`requirements.txt`** includes all the dependencies needed to run the programme.
- **`run.sh`** is the script to automate the execution of main.py, which executes all the scripts in sequence to run the MLP.

## c. Instructions for Executing the Pipeline and Modifying Parameters
1. Clone the repository to your local machine.
2. Make a virtual environment if you have not done so with 'python3 -m venv <venv_name>'
3. Activate your virtual environment using '<venv_name>\Scripts\activate' for windows, or 'source <venv_name>/bin/activate' for macOS/ Linux
4. Ensure you have the required Python dependencies. You can install them using the following command:
'pip install -r requirements.txt'
5. Move the calls.db (agri.db) database into the data folder in src folder (src/data), with temp.db. 
farm_data will be fetched from calls.db, then cleaned and processed and saved into temp.db.
temp.db will be holding the cleaned and processed data frame as well as evaluation metrics for all models used so please do not remove it.
6. Run the pipeline by executing the `run.sh` file in bash:
'./run.sh'
7. It should then run all of the python scripts in this order:
	1.	Data Ingestion and Preprocessing  
	1.1.	Data_Ingestion_And_Preprocessing.py  
	1.2.	Plant_PCA.py  
	2.	Regressors  
	2.1.	Gradient_Boosting_Regressor.py  
	2.2.	Random_Forest_Regressor.py  
	2.3.	Linear_Regression.py  
	3.	Regression Evaluation  
	3.1.	Regression_Evaluation.py  
	4.	Classifiers with Label Encoding  
	4.1.	Decision_Tree_Classifier_Label_Raw.py (Raw Cleaned Data)  
	4.2.	Decision_Tree_Classifier_Label_PCA.py (PCA Cleaned Data)  
	4.3.	K_Nearest_Neighbours_Classifier_Label_PCA.py  
	4.4.	Random_Forest_Classifier_Label_PCA.py  
	5.	Classifiers with OneHot Encoding  
	5.1.	Random_Forest_Classifier_OneHot_PCA.py  
	5.2.	Decision_Tree_Classifier_OneHot_PCA.py  
	5.3.	K_Nearest_Neighbours_Classifier_OneHot_PCA.py  
	6.	Application of Models   
	6.1.    application_test.py
   
This is the sequence in which the scripts are executed based on the run_script() function calls in the main.py script when executing run.sh. Each section (such as Regressors, Classifiers, etc.) is printed before the scripts are run.
7. To modify any parameters, navigate to the specific Python script (e.g., `Data_Ingestion_And_Preprocessing.py`, `Gradient_Boosting_Regressor.py`), and adjust the parameters in the script as needed (e.g., learning rate, n_estimators).
8. To clear the newly made databases, run dropping_tables.py manually to reset all cleaned tables and evaluation metrics. (cd to src folder first if it fails)

## d. Description of Logical Steps/Flow of the Pipeline
1. **Data Ingestion and Preprocessing**: The raw plant sensor data is loaded from the SQLite database, cleaned, and preprocessed.
2. **Feature Engineering**: After the preprocessing step, relevant features are extracted and transformed for model training. This includes handling categorical data (using Label Encoding) and scaling numeric values (using StandardScaler), as well as the use of Principle Component Analysis.
3. **Model Training**: Different machine learning models are trained, such as Gradient Boosting and Decision Tree classifiers/regressors.
4. **Evaluation**: The performance of the models is evaluated using metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), R-squared, and Adjusted R-squared.

## e. Overview of Key Findings from the EDA in Task 1
The Exploratory Data Analysis (EDA) revealed key insights about the dataset:
- **Missing Data**: Certain sensors like the "Humidity Sensor" had significant missing data but a noticeable trend, which was imputed using KNN Imputation. The other columns consisting of rows with NaN values were dropped.
- **Negative Outliers**: Negative values rows for columns such as Temperature Sensor (°C), Light Intensity Sensor (lux) and EC Sensor (dS/m) were dropped.
- **Temperature Correlation**: The key features with the highest correlation to Temperature which we will be utilising for our Temperature Prediction Machine Learning will be: [Light Intensity, Humidity, Nutrient K, CO2 sensor, as well as Plant Type and Plant Stage].
- **Plant Type Correlation**: The key features with the highest correlation to Plant Type are: [Temperature, Light Intensity, CO2, Nutrient N, P, K, and pH].
- **Plant Stage Correlation**: The key features with the highest correlation to Plant Type are: [Light Intensity Sensor, Humidity, O2 Sensor, CO2 Sensor, Nutrient N, P & K]
- **Principle Component Analysis**: It supports our previous hypothesis of key features being Light Intensity, CO2 Sensor, Nutrient N, P and K, as well as giving more feedback about how Temperature, Humidity and pH Sensor might be useful to predict Plant Type-Stage as well. Thus, newly created PCA dataframe will be used for our Plant Type-Stage Prediction Learning Model.
- **Feature Correlation**: Features such as "Nutrient K Sensor (ppm)" and "Nutrient P Sensor (ppm)" were found to be correlated, and PCA was applied to reduce dimensionality and mitigate multicollinearity.


## f. How the Features in the Dataset Are Processed (Summarised in a Table)

## Feature Processing Table

| **Feature**                     | **Data Type** | **Processing Steps**                                                       |
|----------------------------------|---------------|---------------------------------------------------------------------------|
| System Location Code            | Categorical   | No specific processing|
| Previous Cycle Plant Type       | Categorical   | No specific processing|
| Plant Type                       | Categorical   | Title-cased for consistency, label encoded for machine learning |
| Plant Stage                      | Categorical   | Title-cased for consistency, label encoded for machine learning |
| Temperature Sensor (°C)          | Numeric       | Removed negative values and null rows|
| Humidity Sensor (%)              | Numeric       | Imputed using KNN for missing values, no outlier removal|
| Light Intensity Sensor (lux)     | Numeric       | Removed negative values and null rows |
| CO2 Sensor (ppm)                 | Numeric       | No specific processing |
| EC Sensor (dS/m)                 | Numeric       | Removed negative values |
| O2 Sensor (ppm)                 | Numeric       | No specific processing |
| Nutrient N Sensor (ppm)         | Numeric       | Cleaned for non-numeric values (e.g., 'ppm'), removed non-numeric entries and null rows|
| Nutrient P Sensor (ppm)         | Numeric       | Cleaned for non-numeric values (e.g., 'ppm'), removed non-numeric entries and null rows |
| Nutrient K Sensor (ppm)         | Numeric       | Cleaned for non-numeric values (e.g., 'ppm'), removed non-numeric entries and null rows|
| pH Sensor                        | Numeric       | No specific processing |
| Water Level Sensor (mm)         | Numeric       | Removed null rows |
| Plant Type-Stage (combined)     | Categorical   | Combined 'Plant Type' and 'Plant Stage' to form a new column              |
| Labelled Plant Stage             | Numeric   | Label encoded with numerical values                                        |

## g. Explanation of Your Choice of Models for Each Machine Learning Task
For this project, different models were chosen based on their ability to handle the complexity of the dataset:

### Usage of Grid Search Cross Validation for every model

GridSearchCV is a powerful tool in machine learning used for automating the process of hyperparameter tuning. It systematically evaluates all combinations of hyperparameters within a specified grid and uses cross-validation to assess model performance on unseen data, helping to find the best model configuration. By optimizing hyperparameters, GridSearchCV improves model accuracy, generalizability, and performance, while also reducing the risk of overfitting. It is particularly valuable for fine-tuning complex models with multiple hyperparameters, offering an efficient and systematic approach to improving machine learning model results.

### Machine Learning Models for Weak Correlations and Mixed Data

This section describes the reasoning behind using **Gradient Boosting Regressor (GBR)**, **Random Forest Regressor**, and **Linear Regression** models, particularly in the context of weak correlations and datasets with a mix of categorical and numeric features.

#### 1. Linear Regression

- **Good for Strong Linear Relationships**: Linear Regression performs best when there is a linear relationship between the predictors and the target. However, it can struggle with weak correlations, as the linear assumption might not capture the complexity of the data. It is still useful when the correlations are weak but consistent. This model gives a good baseline as to whether the data points have a linear relationship, allowing for better catered choices (GBR and Random Forest Regressor) after evaluating the results from Linear Regression Model.
- **Simple and Fast**: Linear regression is computationally less intensive compared to more complex models like GBR or Random Forest. It is a good starting point, especially when dealing with weak correlations in smaller datasets.
- **Works with Categorical Data via Encoding**: Categorical variables can be included in Linear Regression models by encoding them into numeric values (e.g., using label encoding). However, it might not perform as well as tree-based methods when dealing with categorical variables that interact in non-linear ways with numeric features.
- **Interpretability**: One of the key strengths of Linear Regression is its interpretability. Even with weak correlations, it gives clear insights into the relationship between predictors and the target, although it might not fully capture non-linear interactions.

##### 2. Gradient Boosting Regressor (GBR)

- **Handles Weak Correlations Well**: GBR is an ensemble method that builds models sequentially. It can effectively handle weak correlations by focusing on residuals from previous iterations, capturing more complex relationships between features that might not be immediately apparent.
- **Captures Non-Linear Relationships**: GBR performs well even in cases where there is no clear linear relationship between features and the target variable, which is common when working with weakly correlated data.
- **Handles Mixed Data Types**: GBR can handle both categorical and numeric data, making it an excellent choice when the dataset includes a combination of data types. It allows for flexibility in handling diverse features.
- **Robustness to Overfitting**: With proper tuning of hyperparameters like learning rate and number of estimators, GBR can prevent overfitting, even with weak correlations, by not over-relying on any particular feature.

#### 3. Random Forest Regressor

- **Handles Weak Correlations with High Variability**: Random Forests create multiple decision trees, each trained on a random subset of the data. This ensemble method reduces the impact of weak correlations by averaging the predictions of many models, making it more robust to noise and weak relationships.
- **Works Well with Mixed Data**: Like GBR, Random Forests can naturally handle both categorical and numeric data without needing to explicitly transform the categorical features. This makes it ideal for datasets with mixed types of features.
- **Feature Importance**: Random Forests help identify which features are most predictive, even when some of the features have weak correlations with the target. This is particularly useful when working with many features that have weak or no obvious correlations.
- **Robust to Overfitting**: Random Forests reduce the risk of overfitting by averaging predictions from many different trees, which improves generalization, especially when dealing with weak correlations or noisy data.

#### Summary

- **Linear Regression** is fast and simple but may not fully capture the complexities in the data, especially with weak correlations.
- **GBR** is suitable for datasets with weak correlations and a mix of data types because it can capture non-linear relationships and focus on residual errors from previous models. It is powerful in handling both categorical and numeric features.
- **Random Forest** is robust to weak correlations and works effectively with mixed data types. Its ensemble approach helps to handle data with weak correlations, making it a reliable choice for complex datasets.

By using these models, we can handle datasets with weak correlations between features and the target variable, as well as mixed data types, improving the overall performance and robustness of the model.

### Classification Models and PCA Usage

This section explains why the following classification models — **K-Nearest Neighbors (KNN) Classifier**, **Random Forest Classifier**, and **Decision Tree Classifier** — are chosen for classification tasks. It also explains the use of **PCA (Principal Component Analysis)** for dimensionality reduction, as well as the performance impact of **One-Hot Encoding** and **Label Encoding** on the **target variable** (while predictors remain numeric).

#### 1. K-Nearest Neighbors (KNN) Classifier
KNN Classifier: A non-parametric algorithm that classifies data points based on the majority class of their nearest neighbors in the feature space.

##### Label Encoding (Target Variable)
- **Label Encoding** assigns a unique numerical value to each category in the target variable. This encoding works well for **KNN** as it can directly use the encoded numerical values to compute the distances between data points. 
- **Impact on KNN**: Label encoding works fine with **KNN** if the target variable has an inherent ordinal relationship, where the distances between the labels have meaning (e.g., "low", "medium", "high"). However, for nominal data, label encoding might mislead the model into assuming that there’s an order in the target labels.

##### One-Hot Encoding (Target Variable)
- **One-Hot Encoding** can be used when the target variable is **nominal** (no natural order), where each category is represented as a binary vector. **KNN** will treat these encoded vectors as independent features, which is ideal for non-ordinal data.
- **Impact on KNN**: One-hot encoding allows **KNN** to compute distance between samples without assuming any order in the classes, which is beneficial for categorical targets with no inherent order.

### Why PCA for KNN:
- **Dimensionality Reduction**: PCA is useful in reducing the high-dimensional feature space and removing noise. By selecting the most significant principal components, PCA can help **KNN** perform more efficiently, especially in terms of computational cost, and avoid overfitting due to too many features. 
- **Better Data Point Grouping**:: PCA helps to project the data into a lower-dimensional space where similar data points (e.g., those from the same class) are grouped together more effectively. This can make the classification task easier. This helps classification algorithms, especially those like KNN, decision trees, or logistic regression, to draw clearer decision boundaries between classes and improves the classification accuracy.

---

#### 2. Random Forest Classifier
Random Forest Classifier: An ensemble learning method that builds multiple decision trees and combines their outputs to improve classification accuracy and reduce overfitting.

##### Label Encoding (Target Variable)
- **Label Encoding** works well for **Random Forest** when the target variable is ordinal (i.e., categories that have a clear order). The model can take advantage of the ordinal nature of the target while splitting nodes in the tree.
- **Impact on Random Forest**: **Random Forest** will interpret the numeric values correctly when the target variable is ordinal and can split based on the relative values of the encoded categories.

##### One-Hot Encoding (Target Variable)
- **One-Hot Encoding** is beneficial for **Random Forest** when the target variable is nominal (i.e., no inherent order), as the model will treat each category as a separate entity. It can handle multiple classes effectively using this encoding method.
- **Impact on Random Forest**: **Random Forest** can still perform well with one-hot encoded targets. The trees can make decisions based on each class being distinct, ensuring that each class receives equal weight in the decision-making process.

##### Why PCA for Random Forest:
- **Reducing Feature Complexity**: **Random Forest** benefits from PCA when there are many correlated features. By transforming the original features into orthogonal principal components, **Random Forest** can avoid splitting based on redundant features, leading to better generalization and less overfitting.
  
---

#### 3. Decision Tree Classifier
Decision Tree Classifier: A tree-based algorithm that splits data based on feature values to make decisions, offering interpretability but prone to overfitting without regularization.

##### Label Encoding (Target Variable)
- **Label Encoding** is typically useful for **Decision Trees** when the target variable is ordinal, as it helps the tree to understand the order and make more meaningful splits in the nodes.
- **Impact on Decision Trees**: Label encoding ensures that **Decision Trees** can handle ordinal data correctly by treating the target categories as ordered values when splitting nodes. This is ideal when the target variable has a natural ranking.

##### One-Hot Encoding (Target Variable)
- **One-Hot Encoding** is effective when the target variable is **nominal** and has no natural order. **Decision Trees** can handle one-hot encoded targets by splitting nodes based on the presence or absence of categories, treating each class as a separate entity.
- **Impact on Decision Trees**: One-hot encoding ensures that the tree won't misinterpret the relationships between different categories. Each class is treated distinctly, improving the model’s ability to classify non-ordinal data.

##### Why PCA for Decision Tree:
- **Handling High-Dimensional Data**: **Decision Trees** can overfit if the dataset is high-dimensional with many features. PCA reduces the dimensionality by focusing only on the components with the most variance, which helps **Decision Trees** find clearer and more meaningful splits.
- **Improving Generalization**: By reducing the number of features and removing less important ones, PCA helps **Decision Trees** generalize better to unseen data, improving model performance.

---
### Summary

- **KNN Classifier**: Works well with **Label Encoding** and **One-Hot Encoding** depending on the type of target variable. PCA can improve performance by improving data groupings, reducing feature dimensions and computational cost.
- **Random Forest Classifier**: Handles both **Label Encoding** and **One-Hot Encoding** efficiently. PCA helps by removing redundant features and reducing overfitting.
- **Decision Tree Classifier**: Handles both encodings well, but **Label Encoding** works best for ordinal targets. PCA helps improve decision-making by reducing feature complexity and overfitting.
- Using **PCA** in classification tasks helps reduce the number of features, decrease overfitting, and improve the model's performance by retaining only the most significant features.
---

## h. Evaluation of the Models Developed

### Temperature Regression Model Evaluation Metrics based on [Light Intensity, Humidity, Nutrient K, CO2 sensor, Plant Type & Plant Stage] predictors:

| **Model Name**             | **MSE Test** | **MSE Baseline** | **R²**     | **Adjusted R²** | **MAE**    |
|----------------------------|--------------|------------------|------------|-----------------|------------|
| GradientBoostingRegressor   | 0.890492     | 2.553383         | 0.651072   | 0.650658        | 0.742052   |
| RandomForestRegressor       | 0.885455     | 2.553383         | 0.653046   | 0.652634        | 0.740063   |
| LinearRegression            | 1.370301     | 2.553383         | 0.463065   | 0.462428        | 0.931842   |

#### Evaluation and Analysis

1. **Mean Squared Error (MSE)**: 
   - MSE represents the average squared difference between actual and predicted values. Lower values indicate better performance. Both GradientBoostingRegressor (0.890492) and RandomForestRegressor (0.885455) performed similarly with a much lower MSE than LinearRegression (1.370301), suggesting that the latter is less accurate in this case.

2. **MSE Baseline**: 
   - The baseline MSE is the error obtained when predicting the mean of the target variable for all predictions. It helps to compare model performance. The models are doing significantly better than the baseline (2.553383), as the MSE values are considerably lower.

3. **R-squared (R²)**: 
   - R² represents the proportion of variance in the target variable that is explained by the model. Higher R² values indicate better model performance. Both GradientBoostingRegressor (0.651072) and RandomForestRegressor (0.653046) have relatively high R² values, indicating that they explain more variance in the target variable. On the other hand, LinearRegression has a lower R² of 0.463065, meaning it explains only about 46% of the variance in the target variable.

4. **Adjusted R-squared (Adjusted R²)**:
   - Adjusted R² takes into account the number of predictors in the model and adjusts the R² value accordingly, which helps to prevent overfitting. GradientBoostingRegressor (0.650658) and RandomForestRegressor (0.652634) again show higher values than LinearRegression (0.462428), indicating that these models handle the complexity of the data better.

5. **Mean Absolute Error (MAE)**:
   - MAE represents the average of the absolute differences between predicted and actual values. Lower MAE values indicate better performance. The GradientBoostingRegressor (0.742052) and RandomForestRegressor (0.740063) perform similarly and have lower MAE than LinearRegression (0.931842), suggesting that the latter has a larger average error in its predictions.

#### Conclusion:
The **RandomForestRegressor** and **GradientBoostingRegressor** outperformed **LinearRegression** in almost all evaluation metrics (MSE, R², Adjusted R², and MAE). Both tree-based models performed similarly, indicating that they are well-suited for capturing non-linear relationships in the data. **LinearRegression**, while a simpler model, had a significantly higher MSE and MAE, indicating it struggled with the complexity of the data, especially with the non-linear relationships that the other models captured more effectively.
Thus, the 2 most suitable models for predicting Temperature Conditions are in order: **RandomForestRegressor** and **GradientBoostingRegressor**.


### Plant Type-Stage Prediction Model Evaluation Metrics based on PCA and Raw key features

| **Model Name**                        | **Accuracy** |
|---------------------------------------|--------------|
| DecisionTreeClassifierLabel (Non-PCA) | 0.733320     |
| DecisionTreeClassifierLabel           | 0.769246     |
| KNeighborsClassifierLabel             | 0.813265     |
| RandomForestClassifierLabel           | 0.749901     |
| DecisionTreeClassifierOneHot         | 0.673312     |
| KNeighborsClassifierOneHot           | 0.796881     |

#### Model with Highest Accuracy:
**KNeighborsClassifierLabel**  
**Accuracy:** 0.8133

#### Evaluation and Analysis

#### PCA vs Non-PCA Comparisons:
1. **DecisionTreeClassifierLabel (Non-PCA)**:
   - The **DecisionTreeClassifierLabel (Non-PCA)** model achieved an accuracy of **0.7333**. This model performed reasonably well, but it's likely that the high-dimensionality of the data, including some potentially noisy features, might have hindered its performance. The decision tree model may have struggled with managing the complexity and redundancy in the original feature set.

2. **DecisionTreeClassifierLabel (PCA)**:
   - The **DecisionTreeClassifierLabel (PCA)** model showed improved performance with an accuracy of **0.7692**. The use of **PCA** for dimensionality reduction effectively reduced the feature space, possibly eliminating noise and retaining only the most important features. This reduction in dimensions allowed the decision tree model to focus on the more relevant features, resulting in better accuracy. **PCA** also helped in removing the correlation between features, which might have reduced overfitting and improved generalization.

#### Comparison:
The **PCA-based Decision Tree** model outperformed the **Non-PCA Decision Tree** by a margin, which demonstrates that **PCA** is beneficial in cases where the feature space is large and contains noise. By reducing the dimensionality, **PCA** not only helped speed up the training process but also enhanced the model's ability to generalize better, particularly by reducing overfitting. We will use these findings to support the use of PCA predictors instead of Raw Key Features predictors for the rest of our models.

#### Classifier Models (PCA) Comparisons:
1. **KNeighborsClassifierLabel**:
   - The **KNeighborsClassifierLabel** achieved the highest accuracy of **0.8133**, indicating it was the most effective in classifying the 'Plant Type-Stage'. KNN works by considering the 'k' closest data points, which is likely beneficial in this case where there may be subtle groupings within the data that could help distinguish the target variable. Its performance suggests that relationships between plant types and stages are not too complex for the KNN algorithm to capture, especially when label encoding is used for categorical data.

2. **RandomForestClassifierLabel**:
   - The **RandomForestClassifierLabel** also performed quite well with an accuracy of **0.7499**, but it was outperformed by **KNeighborsClassifierLabel**. Random Forest is an ensemble of decision trees, and it is well-suited for handling non-linearities and complex relationships, which is evident here, although it didn't quite match the performance of KNN.

3. **DecisionTreeClassifierLabel**:
   - **DecisionTreeClassifierLabel** yielded an accuracy of **0.7692**, showing decent performance. Decision trees are easy to interpret but can sometimes overfit, which might have limited its effectiveness compared to KNN, particularly in the context of non-ordinal categories.

4. **KNeighborsClassifierOneHot**:
   - **KNeighborsClassifierOneHot** had an accuracy of **0.7969**, which is good but not as high as **KNeighborsClassifierLabel**. One possible reason is that one-hot encoding introduces a high-dimensional feature space, which might make it harder for the KNN algorithm to efficiently compute distances between data points.

5. **DecisionTreeClassifierOneHot**:
   - The **DecisionTreeClassifierOneHot** performed with an accuracy of **0.6733**, which is lower than the label encoded version of decision trees. The one-hot encoding could have introduced sparsity, and decision trees may have struggled to split the data effectively with such a large number of features.

6. **RandomForestClassifierOneHot**:
	- It was not used and evaluated because it took an extremely long time to complete its training and prediction.

### Conclusion:
The **KNeighborsClassifierLabel** yielded the highest accuracy and speed, outperforming the other models. The use of **label encoding** for categorical variables, despite the target not being ordinal, seems to have worked better than **one-hot encoding**. Label encoding typically maintains the ordinal relationships between categories (even if there isn't a natural order), while one-hot encoding often creates many sparse columns. For models like **KNN**, label encoding allows the algorithm to calculate distances effectively without the need for handling sparse data, which could have been an issue with one-hot encoding. **KNeighborsClassifier** benefits from label encoding as it reduces feature space dimensionality, making the distance calculations more efficient.

In summary, label encoding appears to have provided a more efficient feature space for these classification models, leading to better performance than one-hot encoding, particularly for **KNeighborsClassifier**. However, **RandomForestClassifier** and **DecisionTreeClassifier** struggled with one-hot encoding due to the high dimensionality of the dataset, highlighting that these models may not perform well when dealing with a large number of features created by one-hot encoding.

Thus, the 2 most suitable models for predicting Plant Type-Stage are in order: **KNeighborsClassifier Label/Onehot Encodded** and **DecisionTreeClassifier Label Encodded**.

### Application Test

Using a dummy test sample at which all of the other features apart from Temperature and Plant Type Stage were kept similar, both our Temperature predictor model and Plant Type-Stage predictor model correlated to one another.
With all other features the same and using the Plant Type-Stage of Leafy Green - Seedling, our temperature model gave us a prediction of 23 degree celsius.
With all other features the same and using a temperature similar to the predicted temperature above, 22.8 degree celsius, our Plant Type-Stage model gave us a prediction that the Plant Type-Stage was Leafy Green - Seedling.

Although these are 2 extremely different methods of use of predictors (Raw key feature for temperature and PCA for Plant Type-Stage) and types of prediction model (Random Forest Regressor and KNN Classifier), the results from both models supported one another and proved that both models are reliable at giving good and reliable predictions.

## i. Other Considerations for Deploying the Models Developed
- **Model Deployment**: The models can be deployed as APIs, where users can input new sensor data, and the model returns predictions 
- **Scalability**: Consideration should be given to scaling the models, as sensor data may grow with time, requiring model retraining or online learning.
- **Real-time Data**: For deployment in production environments, real-time data streaming and prediction capabilities should be considered.
