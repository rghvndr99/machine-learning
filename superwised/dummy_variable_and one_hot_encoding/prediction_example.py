#!/usr/bin/env python3
"""
Complete Example: House Price Prediction with Categorical Data
=============================================================

This script demonstrates how to:
1. Load and preprocess data with categorical variables
2. Train a linear regression model
3. Make predictions on new data
4. Handle categorical data properly using scikit-learn

Author: Machine Learning Examples
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_and_explore_data():
    """Load the dataset and explore its structure."""
    print("🏠 House Price Prediction with Categorical Data")
    print("=" * 50)
    
    # Load the data
    df = pd.read_csv('homeprices.csv')
    
    print("Dataset Overview:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
    print(f"\nUnique towns: {df['town'].unique()}")
    print(f"Area range: {df['area'].min()} - {df['area'].max()} sq ft")
    print(f"Price range: ${df['price'].min():,} - ${df['price'].max():,}")
    
    return df

def create_preprocessing_pipeline():
    """Create a preprocessing pipeline for categorical and numerical features."""
    print("\n🔧 Creating Preprocessing Pipeline")
    print("-" * 35)
    
    # Define feature types
    categorical_features = ['town']
    numerical_features = ['area']
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ]
    )
    
    print("✅ Preprocessing pipeline created:")
    print("   - Numerical features: StandardScaler")
    print("   - Categorical features: OneHotEncoder (drop='first')")
    
    return preprocessor

def train_model(df, preprocessor):
    """Train the linear regression model."""
    print("\n🤖 Training Linear Regression Model")
    print("-" * 38)
    
    # Prepare features and target
    X = df[['town', 'area']]
    y = df['price']
    
    # Transform features
    X_transformed = preprocessor.fit_transform(X)
    
    # Train model
    model = LinearRegression()
    model.fit(X_transformed, y)
    
    # Get feature names
    feature_names = (
        preprocessor.named_transformers_['num'].get_feature_names_out(['area']).tolist() +
        preprocessor.named_transformers_['cat'].get_feature_names_out(['town']).tolist()
    )
    
    # Model performance
    r2 = model.score(X_transformed, y)
    predictions = model.predict(X_transformed)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    
    print(f"✅ Model trained successfully!")
    print(f"   R² Score: {r2:.4f}")
    print(f"   RMSE: ${rmse:,.2f}")
    print(f"   Features: {feature_names}")
    
    return model, feature_names, X_transformed, y

def make_predictions(model, preprocessor, feature_names):
    """Demonstrate various prediction scenarios."""
    print("\n🔮 Making Predictions")
    print("-" * 22)
    
    # Example 1: Single prediction
    print("Example 1: Single Prediction")
    new_house = pd.DataFrame({
        'town': ['monroe township'],
        'area': [2800]
    })
    
    transformed = preprocessor.transform(new_house)
    prediction = model.predict(transformed)
    
    print(f"   Input: Monroe Township, 2800 sq ft")
    print(f"   Predicted Price: ${prediction[0]:,.2f}")
    print(f"   Transformed features: {transformed[0]}")
    print()
    
    # Example 2: Multiple predictions
    print("Example 2: Multiple Predictions")
    multiple_houses = pd.DataFrame({
        'town': ['monroe township', 'west windsor', 'robinsville', 'monroe township'],
        'area': [2500, 3500, 2800, 3800]
    })
    
    multiple_transformed = preprocessor.transform(multiple_houses)
    multiple_predictions = model.predict(multiple_transformed)
    
    for i, (town, area, price) in enumerate(zip(
        multiple_houses['town'], 
        multiple_houses['area'], 
        multiple_predictions
    )):
        print(f"   {i+1}. {town.title()}, {area} sq ft → ${price:,.2f}")
    print()
    
    # Example 3: Compare all towns with same area
    print("Example 3: Price Comparison Across Towns (3000 sq ft)")
    comparison_data = pd.DataFrame({
        'town': ['monroe township', 'west windsor', 'robinsville'],
        'area': [3000, 3000, 3000]
    })
    
    comparison_transformed = preprocessor.transform(comparison_data)
    comparison_predictions = model.predict(comparison_transformed)
    
    for town, price in zip(comparison_data['town'], comparison_predictions):
        print(f"   {town.title()}: ${price:,.2f}")
    
    return multiple_predictions

def create_prediction_function(preprocessor, model):
    """Create a reusable prediction function."""
    def predict_house_price(town, area):
        """
        Predict house price given town and area.
        
        Parameters:
        town (str): Town name ('monroe township', 'west windsor', or 'robinsville')
        area (int): House area in square feet
        
        Returns:
        float: Predicted price
        """
        input_data = pd.DataFrame({
            'town': [town.lower()],
            'area': [area]
        })
        
        transformed_data = preprocessor.transform(input_data)
        prediction = model.predict(transformed_data)
        
        return prediction[0]
    
    return predict_house_price

def demonstrate_prediction_function(predict_func):
    """Demonstrate the reusable prediction function."""
    print("\n🎯 Testing Reusable Prediction Function")
    print("-" * 40)
    
    test_cases = [
        ('Monroe Township', 3000),
        ('West Windsor', 3000),
        ('Robinsville', 3000),
        ('Monroe Township', 4000),
        ('West Windsor', 2500)
    ]
    
    for town, area in test_cases:
        price = predict_func(town, area)
        print(f"   {town}, {area} sq ft: ${price:,.2f}")

def analyze_model_insights(model, feature_names):
    """Analyze and explain model coefficients."""
    print("\n📊 Model Insights and Interpretation")
    print("-" * 38)
    
    print(f"Model Intercept: ${model.intercept_:,.2f}")
    print("\nFeature Coefficients:")
    
    for feature, coef in zip(feature_names, model.coef_):
        print(f"   {feature}: {coef:,.2f}")
    
    print("\nInterpretation:")
    print("   - Area coefficient: Price change per sq ft")
    print("   - Town coefficients: Premium/discount vs. reference town")
    print("   - Positive coefficients increase price")
    print("   - Negative coefficients decrease price")
    print("   - Reference town (dropped) is 'monroe township'")

def main():
    """Main execution function."""
    try:
        # Step 1: Load and explore data
        df = load_and_explore_data()
        
        # Step 2: Create preprocessing pipeline
        preprocessor = create_preprocessing_pipeline()
        
        # Step 3: Train model
        model, feature_names, X_transformed, y = train_model(df, preprocessor)
        
        # Step 4: Make predictions
        predictions = make_predictions(model, preprocessor, feature_names)
        
        # Step 5: Create and test prediction function
        predict_func = create_prediction_function(preprocessor, model)
        demonstrate_prediction_function(predict_func)
        
        # Step 6: Analyze model insights
        analyze_model_insights(model, feature_names)
        
        print("\n✅ Example completed successfully!")
        print("You can now use the predict_func to make new predictions.")
        
        return model, preprocessor, predict_func
        
    except FileNotFoundError:
        print("❌ Error: homeprices.csv not found!")
        print("Make sure you're running this script in the correct directory.")
        return None, None, None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None, None

if __name__ == "__main__":
    model, preprocessor, predict_func = main()
