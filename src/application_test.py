import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

## Testing and Application of Regression Model
# Define the column names
columns = ['Light Intensity Sensor (lux)', 'Humidity Sensor (%)', 'Nutrient K Sensor (ppm)', 
           'CO2 Sensor (ppm)', 'Plant Type', 'Labelled Plant Stage']

# Define the values for the new row (these values should correspond to the order of the columns)
new_row = {
    'Light Intensity Sensor (lux)': 144,
    'Humidity Sensor (%)': 69.22,
    'Nutrient K Sensor (ppm)': 168,
    'CO2 Sensor (ppm)': 812,
    'Plant Type': 'Leafy Greens', 
    'Labelled Plant Stage': 1  # Seedling
}
# Convert the dictionary into a DataFrame for the new row
testing_temperature_model_df = pd.DataFrame([new_row], columns=columns)

# Display the new DataFrame with the added row
print(testing_temperature_model_df )

# Load the saved temperature model from the file
loaded_temp_model = joblib.load('random_forest_temperature_regressor_model_trained.pkl')

temperature_prediction = loaded_temp_model.predict(testing_temperature_model_df)
print(temperature_prediction)

## Testing and Application of Classifier Model
# Create a dictionary with the sensor columns
data = {
    'Temperature Sensor (°C)': [22.8],  # Example data
    'Humidity Sensor (%)': [69.22],     # Example data
    'Light Intensity Sensor (lux)': [144], # Example data
    'CO2 Sensor (ppm)': [812],            # Example data
    'EC Sensor (dS/m)': [2.76	],            # Example data
    'O2 Sensor (ppm)': [5],          # Example data
    'Nutrient N Sensor (ppm)': [61],     # Example data
    'Nutrient P Sensor (ppm)': [19],     # Example data
    'Nutrient K Sensor (ppm)': [168],     # Example data
    'pH Sensor': [5.5],                   # Example data
    'Water Level Sensor (mm)': [28]          # Example data
}

# Convert the dictionary into a DataFrame
testing_temperature_model_df = pd.DataFrame(data)

# Load the trained model, scaler, and PCA model
loaded_plant_model = joblib.load('knn_classifier_model_trained.pkl')
scaler = joblib.load('scaler.pkl')  # Load the saved scaler
pca = joblib.load('pca_model.pkl')  # Load the saved PCA model

# Step 1: Apply the same scaling using the loaded scaler
future_data_scaled = scaler.transform(testing_temperature_model_df)

# Step 2: Apply PCA using the loaded PCA model
future_data_pca = pca.transform(future_data_scaled)

# Create a DataFrame with the PCA components and the target variable
plant_pca_df = pd.DataFrame(future_data_pca, columns=[f'PC{i+1}' for i in range(9)])

print(plant_pca_df)

# Step 3: Make predictions using the trained model
plant_type_stage_prediction = loaded_plant_model.predict(plant_pca_df)

print(plant_type_stage_prediction)

# Define the categories as per the original list
categories = [
    'Vine Crops-Maturity', 'Herbs-Maturity', 'Leafy Greens-Seedling',
    'Fruiting Vegetables-Maturity', 'Fruiting Vegetables-Vegetative',
    'Vine Crops-Vegetative', 'Vine Crops-Seedling', 'Herbs-Seedling',
    'Herbs-Vegetative', 'Fruiting Vegetables-Seedling',
    'Leafy Greens-Maturity', 'Leafy Greens-Vegetative'
]

# Create a LabelEncoder
label_encoder = LabelEncoder()

# Fit the encoder on the provided categories without sorting
label_encoder.fit(categories)

# Encode the categories
encoded_labels = label_encoder.transform(categories)

# Create a DataFrame to display the result
encoded_df = pd.DataFrame({
    'Category': categories,
    'Encoded Label': encoded_labels
})

# Display the encoded DataFrame
print(encoded_df)

print("Here, both our models correlate to each other.")
print("Temperature model predicted 23 degree celsius from a plant type stage of Leafy Greens-Seedling, with other variables staying the same")
print("And our classifier predicted a Leafy Green-Seedling Plant Type stage from a temperature around the predicted temperature (22.8 degree celsius), with other variables staying the same")
print("Both models work well!")