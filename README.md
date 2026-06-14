# California Housing Price Prediction

## Overview

This project develops a machine learning regression model to predict California housing prices using demographic, geographic, and socioeconomic data from the California Census dataset.

The project evaluates multiple regression algorithms, applies data preprocessing techniques, performs target transformation, and optimizes model performance through hyperparameter tuning. The final goal is to identify the most accurate and reliable model for predicting median house values and supporting data-driven decision-making in the real estate sector.

---

## Business Understanding

Accurately predicting housing prices is essential for various stakeholders, including:

- Real estate investors
- Property developers
- Financial institutions
- Government agencies
- Urban planners

Reliable price prediction helps stakeholders make informed decisions regarding investment opportunities, property valuation, urban development, and housing market strategies.

This project aims to leverage machine learning techniques to estimate median house values based on housing characteristics, geographic information, and demographic indicators.

---

## Objective

The objectives of this project are:

- Build a machine learning model capable of predicting housing prices accurately
- Compare multiple regression algorithms
- Improve predictive performance through feature engineering and target transformation
- Optimize model parameters using hyperparameter tuning
- Select the best-performing model based on predictive accuracy and generalization ability

---

## Dataset

**Source:**

https://drive.google.com/drive/folders/19YA_f36uGR86hTnZuX-Ech59s3AFzXXo

The dataset is derived from the **1990 California Census** and contains aggregated information at the block-group level.

### Dataset Information

The dataset includes various housing, demographic, and geographic variables.

| Feature | Description |
|----------|------------|
| longitude | Longitude coordinate |
| latitude | Latitude coordinate |
| housing_median_age | Median age of houses |
| total_rooms | Total number of rooms |
| total_bedrooms | Total number of bedrooms |
| population | Population count |
| households | Number of households |
| median_income | Median household income |
| ocean_proximity | Distance category from the ocean |
| median_house_value | Median house value (Target Variable) |

---

## Tools & Technologies

This project was developed using:

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- Pipeline
- ColumnTransformer
- Pickle

---

## Project Workflow

### 1. Data Preprocessing

Several preprocessing techniques were applied to prepare the dataset for machine learning models.

#### Categorical Encoding

The categorical feature:

- `ocean_proximity`

was transformed using:

- OneHotEncoder

#### Feature Scaling

Numerical variables were standardized using:

- StandardScaler

#### Pipeline Integration

A Pipeline and ColumnTransformer were implemented to ensure:

- Consistent preprocessing
- Cleaner workflow
- Reduced risk of data leakage

---

### 2. Train-Test Split

The dataset was divided into:

- Training Set: 80%
- Testing Set: 20%

Configuration:

- Random State: 99

Benefits:

- Fair model evaluation
- Reliable performance assessment
- Better generalization measurement

---

### 3. Model Benchmarking

Nine regression algorithms were evaluated.

#### Models Tested

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. K-Nearest Neighbors (KNN)
5. Decision Tree Regressor
6. Support Vector Regressor (SVR)
7. Random Forest Regressor
8. Gradient Boosting Regressor
9. XGBoost Regressor

#### Validation Method

- 5-Fold Cross Validation

#### Evaluation Metrics

- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

---

### 4. Target Variable Transformation

#### Problem

The target variable (`median_house_value`) exhibited:

- Positive skewness
- Presence of outliers

These characteristics can negatively impact model performance.

#### Solution

A logarithmic transformation was applied using:

```python
log1p()
```

Predictions were converted back using:

```python
expm1()
```

Implementation:

- TransformedTargetRegressor

#### Benefits

- Reduced skewness
- Improved model stability
- Better handling of extreme values
- Enhanced predictive accuracy

---

## Model Performance Summary

### Before Hyperparameter Tuning

| Model | MAPE | MAE |
|---------|---------|---------|
| SVR | ~22.1% | ~41,234 |
| Random Forest | ~23.4% | ~43,184 |
| XGBoost | ~23.8% | ~43,850 |

---

### Hyperparameter Tuning

Hyperparameter tuning was performed to improve model performance and reduce prediction error.

#### Best XGBoost Parameters

```python
learning_rate = 0.11
max_depth = 5
n_estimators = 200
```

---

### After Hyperparameter Tuning

| Model | MAPE | MAE |
|---------|---------|---------|
| Tuned XGBoost | **23.13%** | **42,649** |
| Tuned Random Forest | 23.29% | 42,892 |

---

## Final Model Evaluation

### Best Model: XGBoost Regressor

Performance on the test dataset:

| Metric | Value |
|----------|----------|
| R² Score | 0.82 |
| MAE | ~41,966 |
| MAPE | ~22.95% |
| RMSE | ~61,978 |

---

## Model Insights

### Most Influential Features

The most important predictors identified by the model were:

1. ocean_proximity_INLAND
2. median_income

### Interpretation

#### Ocean Proximity

Property location significantly impacts house value.

Homes located in specific geographic regions, particularly coastal areas, tend to have substantially different prices compared to inland properties.

#### Median Income

Areas with higher household income generally exhibit higher housing prices, indicating strong purchasing power and market demand.

---

## Error Analysis

The model demonstrates:

- Good alignment between predicted and actual values
- Stable performance across most price ranges
- Slight deviations in extremely expensive properties
- Mild heteroskedasticity at higher house values

Overall, prediction errors remain within an acceptable range for real-world housing price estimation.

---

## Why XGBoost?

XGBoost was selected as the final model because it offers several advantages:

- Strong predictive performance
- Ability to capture complex non-linear relationships
- Built-in regularization mechanisms
- Robust handling of tabular datasets
- Efficient training process
- Strong generalization capability

Compared to other algorithms tested, XGBoost provided the best balance between accuracy and robustness.

---

## Business Impact

The developed model can support:

### Real Estate Investors

- Property valuation
- Investment opportunity assessment
- Risk analysis

### Financial Institutions

- Mortgage evaluation
- Lending risk assessment

### Government Agencies

- Housing policy planning
- Urban development strategies

### Property Developers

- Market analysis
- Pricing strategies
- Location selection

---

## Limitations

This project has several limitations:

- Limited geographic variables
- No external economic indicators
- No crime-rate information
- No school-quality indicators
- Slightly lower performance on lower-priced properties

---

## Future Improvements

Potential enhancements include:

### Feature Enrichment

Add:

- Distance to city center
- Public transportation access
- School quality ratings
- Crime statistics
- Environmental indicators

### Advanced Modeling

Explore:

- Stacking Ensembles
- Blending Techniques
- LightGBM
- CatBoost
- Deep Learning Models

### Regional Modeling

Develop separate models for:

- Coastal regions
- Inland regions

to better capture local housing market characteristics.

---

## Model Deployment

The final model can be saved and deployed using Pickle.

```python
import pickle

pickle.dump(estimator, open('California_Housing_XGB.sav', 'wb'))
```

Load model:

```python
model = pickle.load(open('California_Housing_XGB.sav', 'rb'))
```

---

## Repository Structure

```text
california-housing-price-prediction/
│
├── California Housing Price Prediction.ipynb
├── California_Housing_XGB.sav
└── README.md
```

---

## Conclusion

The tuned XGBoost model achieved the strongest overall performance among all evaluated algorithms.

Key results:

- R² Score: 0.82
- MAE: ~41,966
- MAPE: ~22.95%
- RMSE: ~61,978

The model successfully captures the relationship between housing characteristics and property values, explaining approximately 82% of housing price variance while maintaining stable performance on unseen data.

These results demonstrate the effectiveness of machine learning techniques for housing price prediction and provide valuable insights for stakeholders in the real estate industry.
