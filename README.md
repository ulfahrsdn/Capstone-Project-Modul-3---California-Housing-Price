# California Housing Price Prediction

## Business Understanding

Accurately predicting housing prices is essential for stakeholders such as:
- Real estate investors  
- Property developers  
- Financial institutions  
- Government planners  

This project builds a machine learning model to estimate **median house values** based on demographic and geographic features, enabling **data-driven decision making in the housing market**.

---

## Objective

The objective of this project is to:
- Build a robust regression model for predicting housing prices  
- Compare multiple machine learning algorithms  
- Improve performance using transformation and tuning techniques  
- Select the best model based on generalization ability  

---

## Dataset

- Source: https://drive.google.com/drive/folders/19YA_f36uGR86hTnZuX-Ech59s3AFzXXo

The dataset is derived from the **1990 California Census** and contains aggregated data at the block group level.

### Features

| Feature | Description |
|--------|------------|
| longitude, latitude | Location coordinates |
| housing_median_age | Median age of houses |
| total_rooms | Total rooms |
| total_bedrooms | Total bedrooms |
| population | Population count |
| households | Number of households |
| median_income | Median income (scaled) |
| ocean_proximity | Proximity to ocean (categorical) |
| median_house_value | Target variable |

---

## Project Workflow

### 1. Data Preprocessing

- **Categorical Encoding**:
  - OneHotEncoder on `ocean_proximity`
- **Feature Scaling**:
  - StandardScaler for numerical features
- **Pipeline Integration**:
  - ColumnTransformer ensures clean and reproducible preprocessing

---

### 2. Train-Test Split

- Ratio: 80:20  
- Random state: 99  

Ensures:
- Fair evaluation  
- Reliable generalization  

---

### 3. Model Benchmarking

We evaluated 9 regression models:

- Linear Regression  
- Ridge & Lasso  
- KNN  
- Decision Tree  
- SVR  
- Random Forest  
- Gradient Boosting  
- XGBoost  

Using:
- 5-Fold Cross Validation  
- Metrics:
  - MAE
  - MAPE
  - MSE
  - RMSE  

---

### 4. Target Transformation

Problem:
- Target variable is **right-skewed** with outliers  

Solution:
- Apply logarithmic transformation:
  - `log1p()`  
  - `expm1()`  

Tool:
- TransformedTargetRegressor  

Impact:
- Reduced skewness  
- Improved stability  
- Better predictive performance  

---

## Model Performance Summary

### Before Tuning

| Model | MAPE | MAE |
|------|------|------|
| SVR | ~22.1% | ~41,234 |
| Random Forest | ~23.4% | ~43,184 |
| XGBoost | ~23.8% | ~43,850 |

---

### After Tuning

#### Best Model: XGBoost

- learning_rate = 0.11  
- max_depth = 5  
- n_estimators = 200  

| Model | MAPE | MAE |
|------|------|------|
| Tuned XGBoost | **23.13%** | **42,649** |
| Tuned Random Forest | 23.29% | 42,892 |

---

## Final Evaluation (Test Set)

- R² Score: 0.82  
- MAE: ~41,966  
- MAPE: ~22.95%  
- RMSE: ~61,978  

Conclusion:
- Strong generalization  
- No significant overfitting  
- Stable performance  

---

## Model Insights

### Key Drivers

Top features:
1. ocean_proximity_INLAND  
2. median_income  

Interpretation:
- Location strongly influences property value  
- Higher income correlates with higher house prices  

---

### Error Behavior

- Predictions closely follow actual values  
- Slight deviation in high-price range  
- Mild heteroskedasticity detected  

---

## Why XGBoost?

- Handles non-linear relationships effectively  
- Includes built-in regularization  
- Strong performance on structured/tabular data  

---

## Limitations

- Limited spatial features  
- No external socioeconomic data  
- Slightly weaker performance on low-price houses  

---

## Future Improvements

- Add more features:
  - Distance to city center  
  - Public facilities  
  - Environmental indicators  
- Apply advanced ensemble methods (stacking/blending)  
- Explore deep learning models  
- Segment models by region (coastal vs inland)  

---

## Model Deployment

Save model:

```python
import pickle
pickle.dump(estimator, open('California Housing Price_XGB.sav', 'wb'))
