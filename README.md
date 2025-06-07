# Machine Learning Examples

A collection of practical machine learning examples focusing on linear regression implementations using Python, scikit-learn, pandas, and Jupyter notebooks.

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Projects](#projects)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Running the Examples](#running-the-examples)
- [Data Sources](#data-sources)

## 🎯 Overview

This repository contains hands-on machine learning examples that demonstrate various aspects of linear regression:

- **Simple Linear Regression**: Single-variable prediction models
- **Multiple Linear Regression**: Multi-variable prediction models
- **Data Preprocessing**: Handling missing values and data cleaning
- **Model Persistence**: Saving and loading trained models using different methods
- **Real-world Applications**: Home price prediction and economic data analysis

## 📁 Repository Structure

```
machine-learning/
├── linear-reg/                     # Basic linear regression examples
│   ├── linear_reg_home/           # Home price prediction (single variable)
│   └── linear_rqg_capital/        # Canada per capita income analysis
├── linear-reg-multivalue/         # Multiple linear regression
├── dummy_variable_and one_hot_encoding/  # Categorical data preprocessing
│   ├── homeprices.csv             # Home prices with categorical town data
│   ├── carprices.csv              # Car prices with categorical model data
│   ├── pandas_dummy-variable_method-Copy1.ipynb  # Pandas get_dummies approach
│   └── pandas_sklearn_onehot_encoding_method.ipynb  # Scikit-learn approach
├── save-linear-reg-model/         # Model persistence examples
│   ├── save_linear_reg_home_with_joblib/  # Using joblib for model saving
│   └── save_linear_reg_home_with_pickle/  # Using pickle for model saving
└── README.md
```

## 🚀 Projects

### 1. Basic Linear Regression (`linear-reg/`)

#### Home Price Prediction (`linear_reg_home/`)
- **Dataset**: `homeprices.csv` - Contains area and price data
- **Objective**: Predict house prices based on area (square footage)
- **Features**: Single variable (area) linear regression
- **Notebook**: `linear-reg_home_area.ipynb`

#### Canada Per Capita Income Analysis (`linear_rqg_capital/`)
- **Dataset**: `canada_per_capita_income.csv` - Historical income data from 1970-2016
- **Objective**: Analyze income trends over time
- **Features**: Time series linear regression
- **Notebook**: `linear-reg-net-capital.ipynb`

### 2. Multiple Linear Regression (`linear-reg-multivalue/`)
- **Dataset**: `homeprices.csv` - Enhanced with multiple features (area, bedrooms, age)
- **Objective**: Predict house prices using multiple variables
- **Features**:
  - Multiple variable regression
  - Missing data handling (NaN values in bedrooms)
  - Data imputation using median values
- **Notebook**: `linear_m_value.ipynb`

### 3. Dummy Variables and One-Hot Encoding (`dummy_variable_and one_hot_encoding/`)

Comprehensive examples of handling categorical data in machine learning:

#### Pandas Dummy Variables Method (`pandas_dummy-variable_method-Copy1.ipynb`)
- **Dataset**: `homeprices.csv` - Home prices with town categories
- **Method**: Using pandas `get_dummies()` function
- **Features**:
  - Simple categorical encoding
  - Avoiding dummy variable trap
  - Concatenating dummy variables with original data
- **Use Case**: Quick and simple categorical encoding for pandas workflows

#### Scikit-learn One-Hot Encoding Method (`pandas_sklearn_onehot_encoding_method.ipynb`)
- **Dataset**: `homeprices.csv` and `carprices.csv`
- **Method**: Using scikit-learn's `OneHotEncoder` with `ColumnTransformer`
- **Features**:
  - Modern preprocessing pipeline approach
  - Proper separation of categorical and numerical features
  - Feature scaling integration
  - Production-ready preprocessing
  - Avoiding multicollinearity with `drop='first'`
- **Recently Fixed**: Updated to use current scikit-learn API (removed deprecated `categorical_features` parameter)

### 4. Model Persistence (`save-linear-reg-model/`)

Demonstrates two different approaches to saving and loading trained models:

#### Using Joblib (`save_linear_reg_home_with_joblib/`)
- **Method**: joblib library for model serialization
- **Advantages**: Efficient for NumPy arrays and scikit-learn models
- **Output**: `model_joblib` file

#### Using Pickle (`save_linear_reg_home_with_pickle/`)
- **Method**: Python's built-in pickle module
- **Advantages**: Standard Python serialization
- **Output**: `model_pickle` file

## 🛠 Technologies Used

- **Python 3.x**
- **Jupyter Notebook** - Interactive development environment
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **scikit-learn** - Machine learning library
  - `LinearRegression` - Linear regression models
  - `OneHotEncoder` - Categorical data encoding
  - `ColumnTransformer` - Feature preprocessing pipelines
  - `StandardScaler` - Feature scaling
  - `LabelEncoder` - Label encoding (legacy approach shown)
- **matplotlib** - Data visualization
- **joblib** - Model persistence
- **pickle** - Object serialization

## 🏁 Getting Started

### Prerequisites

Make sure you have Python 3.x installed along with the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib jupyter joblib
```

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd machine-learning
```

2. Launch Jupyter Notebook:
```bash
jupyter notebook
```

## 🎮 Running the Examples

### Basic Linear Regression
1. Navigate to `linear-reg/linear_reg_home/`
2. Open `linear-reg_home_area.ipynb`
3. Run all cells to see home price prediction based on area

### Multiple Linear Regression
1. Navigate to `linear-reg-multivalue/`
2. Open `linear_m_value.ipynb`
3. Run all cells to see multi-variable home price prediction

### Categorical Data Preprocessing

#### Pandas Dummy Variables Method
1. Navigate to `dummy_variable_and one_hot_encoding/`
2. Open `pandas_dummy-variable_method-Copy1.ipynb`
3. Run all cells to see how pandas `get_dummies()` handles categorical data
4. Learn about avoiding the dummy variable trap

#### Scikit-learn One-Hot Encoding (Recommended)
1. Navigate to `dummy_variable_and one_hot_encoding/`
2. Open `pandas_sklearn_onehot_encoding_method.ipynb`
3. Run all cells to see modern preprocessing with `ColumnTransformer`
4. This approach is production-ready and handles both categorical and numerical features properly

### Model Persistence
1. Navigate to either `save-linear-reg-model/save_linear_reg_home_with_joblib/` or `save-linear-reg-model/save_linear_reg_home_with_pickle/`
2. Open the respective notebook
3. Run all cells to train a model and save it
4. The saved model files can be loaded later for predictions

## 📊 Data Sources

### Home Prices Dataset (Multiple Versions)
- **Basic Version**: Area (sq ft), Price - for simple linear regression
- **Extended Version**: Area (sq ft), Bedrooms, Age, Price - for multiple regression
- **Categorical Version**: Town, Area (sq ft), Price - for categorical data examples
- **Use Case**: Real estate price prediction with different complexity levels
- **Format**: CSV with headers

### Canada Per Capita Income Dataset
- **Features**: Year (1970-2016), Per Capita Income (US$)
- **Use Case**: Economic trend analysis
- **Format**: CSV with headers

### Car Prices Dataset
- **Features**: Car Model, Mileage, Sell Price ($), Age (years)
- **Use Case**: Vehicle price prediction with categorical car models
- **Format**: CSV with headers
- **Categories**: BMW X5, Audi A5, Mercedes Benz C class

## 🎓 Learning Outcomes

By working through these examples, you'll learn:

- How to implement linear regression using scikit-learn
- Data preprocessing techniques for real-world datasets
- Handling missing values in datasets
- Multiple linear regression with several features
- **Categorical data preprocessing**:
  - Converting categorical variables to numerical format
  - Using pandas `get_dummies()` for simple cases
  - Using scikit-learn's `OneHotEncoder` for production pipelines
  - Understanding and avoiding the dummy variable trap
  - Combining categorical and numerical feature preprocessing
- **Modern ML preprocessing pipelines**:
  - Using `ColumnTransformer` for different feature types
  - Feature scaling with `StandardScaler`
  - Building reusable preprocessing pipelines
- Model evaluation and prediction
- Different methods for model persistence
- Data visualization with matplotlib
- Working with Jupyter notebooks for ML development
- **Best practices for handling mixed data types in machine learning**

## 🤝 Contributing

Feel free to contribute by:
- Adding new machine learning examples
- Improving existing notebooks
- Adding more comprehensive documentation
- Suggesting new datasets or use cases

## 📝 License

This project is open source and available under the [MIT License](LICENSE).