# ⚡ Power Consumption Optimization in Zone 1 - Wellington, New Zealand  

## 📝 Problem Statement  

Wellington, New Zealand faces fluctuations in power demand due to environmental and meteorological changes. To ensure sustainable energy consumption and efficient resource planning, we aim to build a predictive model that estimates **Zone 1 Power Consumption** based on factors like temperature, humidity, wind speed, cloudiness, air quality, and solar radiation flows.  

---  

## 🎯 Objective  

To develop a **Machine Learning model** that:
- 🔍 Predicts **Zone 1 Power Consumption**
- Uses environmental features like:  
   🌡️ Temperature  
   💧 Humidity  
   🌬️ Wind Speed  
   ☀️ General Diffuse Flows  
   🌥️ Diffuse Flows  
   🏭 Air Quality Index (PM)  
   ☁️ Cloudiness  
- ⚡ Helps in optimizing energy usage, reducing cost, and ensuring sustainability.

---  

## 📁 Dataset Description  

| Column                         | Description                                                                                          |
|--------------------------------|------------------------------------------------------------------------------------------------------|
| 🔢 `Sr no.`                    | Serial Number                                                                                        |
| 🌡️ `Temperature`               | The temperature in Celsius at the specific location.                                                 |
| 💧 `Humidity`                  | The relative humidity percentage at the location (g/m³ – grams of water vapor per cubic meter).     |
| 🌬️ `Wind Speed`                | The speed of the wind at the location (nautical miles per hour).                                     |
| ☀️ `General Diffuse Flows`     | Refers to the amount/intensity of diffuse solar radiation in a specific area (m²/s).                 |
| 🌥️ `Diffuse Flows`             | The measure of diffuse solar radiation (m²/s).                                                       |
| 🏭 `Air Quality Index (PM)`    | An index representing air quality in the area (particles in micrograms per cubic meter).             |
| ☁️ `Cloudiness`                | The level of cloud cover at the location (1 = Yes, 0 = No).                                          |
| ⚡ `Power Consumption in A Zone` | The power consumption in Zone 1 (target variable) measured in kilowatt rating (KWR).                |

---  

## 🛠️ Tools & Technologies

- **Programming Language:** Python  
- **Data Manipulation:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Modeling & Evaluation:**  
  - Scikit-learn (Linear Regression, Decision Tree, Random Forest, SVR)  
  - XGBoost  
- **Environment:** Jupyter Notebook  
- **Other Techniques:**  
  - Feature Engineering  
  - Outlier Capping  
  - Skewness Reduction  
  - Correlation Heatmap Analysis  

---  

## 📚 Project Workflow  

### 1️⃣ Import Necessary libraries  
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta, datetime

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")
```
---  

### 2️⃣ Data Loading and Preview

![Screenshot 2025-07-07 163958](https://github.com/user-attachments/assets/a7ce64a6-db8c-4291-9083-38e7e4fefca9)  

---  

### 3️⃣ Data Overview & Cleaning  

The dataset contained **52,583 records** and **9 columns**.  

**🔹Converted Object Columns to Numeric**
- `Temperature` and `Humidity` columns were initially stored as objects.  
- Used `pd.to_numeric(..., errors='coerce')` to convert them to `float64`.
  
**🔹Renamed Columns for Consistency**  
- Replaced spaces and special characters with underscores using:  
```  
df.rename(columns=lambda x: x.strip().replace(" ", "_").replace("(", "").replace(")", ""), inplace=True)
```

**🔹Dropped Irrelevant Columns**  
- Removed the serial number column `S_no`.  

**🔹Missing Value Treatment**  
- Missing values found in multiple features (e.g., Temperature: 323 missing).  
- Used **median imputation** to fill missing values:
```
df[col] = df[col].fillna(df[col].median())
```

**🔹Duplicate Check**  
- Found **0 duplicated rows** in the dataset.

---  

### 4️⃣ Exploratory Data Analysis (EDA)  

#### Univariate Analysis  

![Univariate_one](https://github.com/user-attachments/assets/e9766e64-578b-41a7-97e7-5de58662bd11)  

![Univariate_two](https://github.com/user-attachments/assets/abcc5e1d-031d-45cb-b31c-766e16210120)
  
---  

#### Bivariate Analysis  

![Bivariate_one](https://github.com/user-attachments/assets/0762616a-9c66-40fc-95ff-5eafeb99e7e5)  
**Interpreting the Red Line in Regression Plots**

The **red line** in each regression plot shows the **trend** between the feature and power consumption. Here's how to interpret it:  

| **Red Line Trend**             | **Meaning**                                                  |
|--------------------------------|----------------------------------------------------------------|
| 📈 **Upward Sloping**          | As the feature increases, power consumption **increases** → *Positive correlation* |
| 📉 **Downward Sloping**        | As the feature increases, power consumption **decreases** → *Negative correlation* |
| ➖ **Flat or No Slope**        | No clear relationship between the feature and power consumption → *Weak or zero correlation* |

This helps in understanding how strongly and in what direction each feature influences power consumption.  



![Bivariate_two](https://github.com/user-attachments/assets/c9643065-28e2-4bfe-b709-92a101c2a7fb)  
**Insight: Impact of Cloudiness on Power Consumption in A Zone**  
Both the Boxplot and Violin Plot clearly indicate that:  
- Power consumption is higher on clear days (Cloudiness = 0) compared to cloudy days (Cloudiness = 1).
- The median power consumption on clear days is significantly higher than on cloudy days.
- The range and variability in power consumption are also larger on clear days, suggesting more extreme energy usage.
- Violin plot shows denser concentration of high consumption values on clear days, while cloudy days have a tighter and lower distribution.

---  

#### Multivariate Analysis  

![Multivariate](https://github.com/user-attachments/assets/1c2c62a8-1d0c-4918-9168-e6c08a9546e6)  

---  

#### Outlier Detection  

![All_Outliers_Boxplots](https://github.com/user-attachments/assets/9d32ef1c-161d-4f41-87d7-fc3f2ccb5f7f)

---  

#### Skewness Reduction  

![Skewness](https://github.com/user-attachments/assets/9d60bc1f-ba39-4bbd-9ff9-40e1ce292186)  

---  

#### Correlation Heatmap  

![Heatmap](https://github.com/user-attachments/assets/690cf4b8-0040-4c54-899b-3ab20ecf4e98)  

**Insights from Heatmap**  
- `Temperature` has the strongest positive correlation (+0.56) with power consumption.  
- `diffuse_flows` and `general_diffuse_flows` show very high correlation (+0.96) → indicating redundancy.  
- `Air_Quality_Index_PM` and `Cloudiness` show almost no correlation with the target → low predictive value.
- Based on these insights, we dropped:
```
df.drop(['Air_Quality_Index_PM', 'diffuse_flows', 'Cloudiness'], axis=1, inplace=True)
```
This helped improve model efficiency, reduce overfitting, and avoid multicollinearity.  

---  

### 5️⃣ Train-Test Split  

- The dataset was divided into features (`X`) and target (`y`) where:  
   `X` = Environmental features  
   `y` = Power Consumption in A Zone
- Used `train_test_split()` to divide data:  
  80% for training, 20% for testing  
  Ensured reproducibility using `random_state=42`
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- Training Shape: 42,066 samples
- Testing Shape: 10,517 samples

---  

### 6️⃣ Feature Scaling  

- Applied StandardScaler to scale numerical features:  
`Temperature`, `Humidity`, `Wind_Speed`, and `general_diffuse_flows`  
- Fitted the scaler **only on training data** to avoid data leakage.
```
scaler = StandardScaler()
X_train_scaled[features] = scaler.fit_transform(X_train[features])
X_test_scaled[features] = scaler.transform(X_test[features])
```
This ensured that all selected features were normalized to mean 0 and variance 1 for better model performance.  

---  

### 7️⃣ Model Building  
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor
- Support Vector Regressor (SVR)

---  

### 📊 Model Performance Comparison Before Hyperparameter Tuning  

| Model                    | Training Accuracy (R²) | Testing Accuracy (R²) | MAE       | MSE           | RMSE     |
|--------------------------|------------------------|------------------------|-----------|----------------|----------|
| Linear Regression        | 0.3181                 | 0.3252                 | 5310.4315 | 43589314.9753 | 6602.2204 |
| Decision Tree Regressor | 0.9982                 | 0.2671                 | 4205.5229 | 47340547.6567 | 6880.4467 |
| Random Forest Regressor | 0.9448                 | 0.6192                 | 3447.8083 | 24598744.4930 | 4959.7121 |
| XGBoost Regressor       | 0.6439                 | 0.5485                 | 4027.3442 | 29160553.7507 | 5400.0512 |
| Support Vector Regressor| 0.1412                 | 0.1466                 | 5921.9808 | 55125072.1834 | 7424.6260 |  


**Insights from R² Score Comparison**  

- Based on this comparison, **Random Forest** and **XGBoost** are **strong candidates for final model selection**. Therefore, further **hyperparameter tuning will be applied only on these two models** to improve their performance.

---  

### 📊 Model Performance Comparison After Hyperparameter Tuning 

| Model                    | Training Accuracy (R²) | Testing Accuracy (R²) | MAE       | MSE           | RMSE     |
|--------------------------|------------------------|------------------------|-----------|----------------|----------|
| Linear Regression        | 0.3181                 | 0.3252                 | 5310.4315 | 43589314.9753 | 6602.2204 |
| Decision Tree Regressor | 0.9982                 | 0.2671                 | 4205.5229 | 47340547.6567 | 6880.4467 |
| Random Forest Regressor | 0.9099                 | 0.6205                 | 3475.5517 | 24512550.1897 | 4951.0150 |
| XGBoost Regressor       | 0.8062                 | 0.5943                 | 3692.2429 | 26202533.6600 | 5118.8410 |
| Support Vector Regressor| 0.1412                 | 0.1466                 | 5921.9808 | 55125072.1834 | 7424.6260 |  
 
🏆 **Best Model: Random Forest** (better performance on all evaluation metrics)  
**Reason for Selection:**  
- **Highest Testing Accuracy (R²):** `0.6205`  
- **Lowest MAE:** `3475.55`  
- **Lowest MSE:** `24512550.18`  
- **Lowest RMSE:** `4951.01`

---  

## ⚠️ Limitations

- **Moderate R² Score:**  
  Although the Random Forest model had the best performance, an R² score of 0.62 indicates that a portion of the variability in power consumption remains unexplained.

- **No Temporal Features Included:**  
  Features such as time of day, date, or seasonality were not included, which limits the model's ability to capture cyclical trends in electricity usage.

- **Static Dataset Used:**  
  The model was trained on historical data and doesn't currently adapt to live or changing conditions in real-time.

- **Outlier Capping and Skewness Correction Were Basic:**  
  While outlier treatment and skewness correction were performed, more advanced techniques (like IQR-based adaptive capping or Box-Cox transformation) could yield better results.

- **External Factors Not Considered:**  
  The model does not account for socio-economic or policy-driven factors that might also influence power consumption (e.g., holidays, industrial activity).

---  

## 🔚 Conclusion  

This project aimed to predict **Zone 1 power consumption in Wellington, New Zealand** using environmental and meteorological factors, in order to support efficient and sustainable energy management.

### 📌 Summary:

1. **Data Cleaning & Preparation**
   - Processed a dataset of **52,583 records**
   - Converted object columns like `Temperature` and `Humidity` to numeric
   - Handled missing values using **median imputation**
   - Dropped irrelevant column `S_no`  

2. **Exploratory Data Analysis & Feature Selection**
   - Detected and **capped outliers** in numerical columns to reduce the influence of extreme values
   - Applied **skewness reduction** to normalize heavily skewed distributions
   - Used a correlation heatmap to detect multicollinearity and weak features
   - Dropped features: `Air_Quality_Index_PM`, `Cloudiness`, and `diffuse_flows`
   - Found `Temperature` to be strongly correlated with power consumption (+0.56)  

4. **Model Building & Evaluation**
   - Trained five regression models:  
     `Linear Regression`, `Decision Tree`, `Random Forest`, `XGBoost`, `SVR`
   - Evaluated using: **R² Score, MAE, MSE, RMSE**
   - Since the goal was to reduce prediction error, the model was selected **based on lowest MAE and RMSE**
   - **Best Model: Random Forest Regressor**
     - R² Score: `0.6205`
     - MAE: `3475.55`
     - MSE: `24512550.18`
     - RMSE: `4951.01`  

5. **Feature Importance (from Random Forest)**
   - `Temperature`: **0.50**
   - `Humidity`: **0.21**
   - `general_diffuse_flows`: **0.16**
   - `Wind_Speed`: **0.13**  

### 🎯 Final Outcome:
The Random Forest model provided the most accurate and consistent predictions. It effectively captures the influence of environmental variables on energy usage and can be used to:  
- Forecast power demand in Zone 1
- Minimize energy waste
- Support real-time and sustainable energy planning
  
---  

## 🧑‍💻 Author

**Ashwini Bawankar**  
*Data Science Intern | Passionate about Machine Learning*

---

## 📬 Contact

📧 Email: [abawankar13@gmail.com]  
🔗 LinkedIn: [https://www.linkedin.com/in/ashwini-bawankar/]  




  




















































































