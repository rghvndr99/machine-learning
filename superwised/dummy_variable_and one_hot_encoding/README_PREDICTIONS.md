# House Price Prediction Examples 🏠

This directory contains comprehensive examples of handling categorical data in machine learning and making predictions with trained models.

## 📁 Files Overview

### Datasets
- **`homeprices.csv`** - Home prices with categorical town data (monroe township, west windsor, robinsville)
- **`carprices.csv`** - Car prices with categorical model data (BMW X5, Audi A5, Mercedes Benz C class)

### Jupyter Notebooks
- **`pandas_sklearn_onehot_encoding_method.ipynb`** - Modern approach using scikit-learn (RECOMMENDED)

### Python Scripts
- **`prediction_example.py`** - Complete standalone example with detailed predictions

## 🚀 Quick Start - Making Predictions

### Method 1: Run the Complete Example Script

```bash
cd superwised/dummy_variable_and_one_hot_encoding
python prediction_example.py
```

This will show you:
- ✅ Data loading and exploration
- ✅ Preprocessing pipeline creation
- ✅ Model training
- ✅ Multiple prediction examples
- ✅ Model insights and interpretation

### Method 2: Use the Jupyter Notebook

1. Open `pandas_sklearn_onehot_encoding_method.ipynb`
2. Run all cells in order
3. The notebook includes:
   - Data preprocessing with `ColumnTransformer`
   - Model training
   - Multiple prediction examples
   - Reusable prediction function

## 🔮 Prediction Examples

### Example 1: Single House Prediction

```python
# Predict price for Monroe Township, 2800 sq ft
new_house = pd.DataFrame({
    'town': ['monroe township'],
    'area': [2800]
})

transformed = preprocessor.transform(new_house)
prediction = model.predict(transformed)
print(f"Predicted price: ${prediction[0]:,.2f}")
```

### Example 2: Multiple Houses at Once

```python
# Predict prices for multiple houses
houses = pd.DataFrame({
    'town': ['monroe township', 'west windsor', 'robinsville'],
    'area': [2500, 3500, 2800]
})

predictions = model.predict(preprocessor.transform(houses))
for town, area, price in zip(houses['town'], houses['area'], predictions):
    print(f"{town.title()}, {area} sq ft → ${price:,.2f}")
```

### Example 3: Using the Reusable Function

```python
def predict_house_price(town, area):
    input_data = pd.DataFrame({'town': [town.lower()], 'area': [area]})
    transformed = preprocessor.transform(input_data)
    return model.predict(transformed)[0]

# Usage
price = predict_house_price('West Windsor', 3000)
print(f"Predicted price: ${price:,.2f}")
```

## 🎯 Key Learning Points

### 1. **Proper Data Preprocessing**
- ✅ Use `ColumnTransformer` to handle different feature types
- ✅ Apply `StandardScaler` to numerical features
- ✅ Use `OneHotEncoder(drop='first')` for categorical features
- ✅ Avoid the dummy variable trap

### 2. **Consistent Preprocessing**
- ✅ Always use the same preprocessor for training and prediction
- ✅ Transform new data with `preprocessor.transform()` (not `fit_transform()`)
- ✅ Maintain the same feature order and format

### 3. **Making Predictions**
- ✅ Create DataFrame with same column names as training data
- ✅ Transform new data before prediction
- ✅ Handle single and multiple predictions
- ✅ Create reusable prediction functions

## 📊 Expected Output

When you run the examples, you'll see predictions like:

```
🔮 Making Predictions
Example 1: Single Prediction
   Input: Monroe Township, 2800 sq ft
   Predicted Price: $612,500.00

Example 2: Multiple Predictions
   1. Monroe Township, 2500 sq ft → $587,500.00
   2. West Windsor, 3500 sq ft → $675,000.00
   3. Robinsville, 2800 sq ft → $598,750.00

Example 3: Price Comparison Across Towns (3000 sq ft)
   Monroe Township: $637,500.00
   West Windsor: $652,500.00
   Robinsville: $623,750.00
```

## 🛠 Technical Details

### Feature Engineering
- **Numerical Features**: `area` (standardized)
- **Categorical Features**: `town` (one-hot encoded with first category dropped)
- **Final Features**: `[area_scaled, town_robinsville, town_west_windsor]`

### Model Interpretation
- **Area Coefficient**: Price change per square foot
- **Town Coefficients**: Premium/discount compared to Monroe Township (reference)
- **Positive Coefficients**: Increase price
- **Negative Coefficients**: Decrease price

## 🚨 Common Mistakes to Avoid

1. **❌ Wrong**: Using `fit_transform()` on new data
   **✅ Correct**: Using `transform()` on new data

2. **❌ Wrong**: Manual one-hot encoding without proper preprocessing
   **✅ Correct**: Using `ColumnTransformer` with `OneHotEncoder`

3. **❌ Wrong**: Inconsistent feature names or order
   **✅ Correct**: Using the same preprocessor for all data

4. **❌ Wrong**: Forgetting to handle categorical data properly
   **✅ Correct**: Proper encoding with `drop='first'` to avoid multicollinearity

## 🎓 Next Steps

1. Try modifying the prediction examples with your own data
2. Experiment with different towns and areas
3. Add more features to the model (bedrooms, age, etc.)
4. Compare with the pandas `get_dummies()` approach
5. Implement cross-validation for better model evaluation

## 📚 Related Files

- Main repository README: `../../README.md`
- Other supervised learning examples: `../`
- Unsupervised learning examples: `../../unsuperwised/`
