# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
import joblib
from scipy import stats

# Load the data
data = pd.read_excel('DATA AR JANUARI - MEI 2023 - FIX - Copy.xlsx')
data.head()

# Feature Engineering
data2 = data[(data.Bandwidth != "Units") & (data.Layanan != "Penambahan IPv4 Publik Internet Corporate") & (data.Biaya_Sewa != 0.0) & 
             (data.AL_Status == "Active") & (data.SBU_Ter == "SBU REG JAWA BAGIAN TIMUR") & (data.AR_Type == "New")]

data3 = data2.drop(['Nomor', 'Segment', 'Cust Name', 'Address Terminating', 'SBU_Ter', 'AL_Status', 'AR_Type'], axis=1)

# Preprocessing
# Identify categorical columns
categorical_columns = ["Bidang_Baku", "Tipe", "Layanan", "Kabupaten/Kota", "Wilayah"]

# Define ordinal encoder
encoder = OrdinalEncoder()

# Perform ordinal encoding
data3[categorical_columns] = encoder.fit_transform(data3[categorical_columns])

# Convert from float to int
for col in categorical_columns + ["Bandwidth", "Biaya_Sewa"]:
    data3[col] = data3[col].astype(int)

# Fill NaN values with 0
data3 = data3.fillna(0)

# Check for outliers
numerical_vars = ['Bidang_Baku', 'Layanan', 'Bandwidth', 'Biaya_Sewa', 'Kabupaten/Kota', 'Wilayah']
outliers = []
for var in numerical_vars:
    z_scores = stats.zscore(data3[var])
    threshold = 3  # Adjust this threshold as needed
    var_outliers = data3[np.abs(z_scores) > threshold]
    outliers.append(var_outliers)
outliers = pd.concat(outliers)
print("Outliers:")
print(outliers)

# Split the data into training and testing sets
X = data3.drop(['Layanan', 'Biaya_Sewa'], axis=1)
y = data3[['Layanan', 'Biaya_Sewa']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Create a multi-output regressor using BaggingRegressor
regressor = MultiOutputRegressor(BaggingRegressor())

# Fit the model on the training data
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_test_pred_mo = regressor.predict(X_test)
y_train_pred_mo = regressor.predict(X_train)

# Calculate the Mean Squared Error of the model
train_mse = mean_squared_error(y_train, y_train_pred_mo, multioutput='raw_values')
test_mse = mean_squared_error(y_test, y_test_pred_mo, multioutput='raw_values')

print("Train Set Mean Squared Error:", train_mse)
print("Test Set Mean Squared Error:", test_mse)

# Save the trained model as a .pkl file
joblib.dump(regressor, 'rmo_model.pkl')