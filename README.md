# Machine Learning Examples

A comprehensive collection of machine learning examples organized by learning type, covering supervised and unsupervised learning algorithms with practical implementations using Python, scikit-learn, pandas, and Jupyter notebooks.

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Supervised Learning](#supervised-learning)
- [Unsupervised Learning](#unsupervised-learning)
- [Advanced Topics](#advanced-topics)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Quick Start Examples](#quick-start-examples)

## 🎯 Overview

This repository contains hands-on machine learning examples organized into two main categories:

### **Supervised Learning**
- **Linear Regression**: Single and multiple variable models
- **Logistic Regression**: Classification problems
- **Decision Trees & Random Forest**: Tree-based algorithms
- **Support Vector Machines**: SVM for classification and regression
- **K-Nearest Neighbors**: Instance-based learning
- **Model Evaluation**: Cross-validation, train/test splits

### **Unsupervised Learning**
- **K-Means Clustering**: Centroid-based clustering
- **Naive Bayes**: Probabilistic classification
- **Principal Component Analysis (PCA)**: Dimensionality reduction

### **Advanced Topics**
- **Regularization**: L1 & L2 regularization techniques
- **Hyperparameter Tuning**: Grid search and optimization
- **Ensemble Learning**: Combining multiple models
- **Outlier Detection**: IQR-based outlier removal

## 📁 Repository Structure

```
machine-learning/
├── superwised/                     # Supervised Learning Examples
│   ├── linear-reg/                # Linear regression examples
│   ├── linear-reg-multivalue/     # Multiple linear regression
│   ├── dummy_variable_and one_hot_encoding/  # Categorical data preprocessing
│   │   ├── homeprices.csv         # Home prices with categorical data
│   │   ├── carprices.csv          # Car prices with categorical data
│   │   ├── pandas_sklearn_onehot_encoding_method.ipynb  # Modern approach
│   │   ├── prediction_example.py  # Complete prediction example
│   │   └── README_PREDICTIONS.md  # Detailed prediction guide
│   ├── logistic_regression/       # Classification examples
│   ├── logistic_regression-multivalue/  # Multi-class classification
│   ├── decesion_tree/            # Decision tree algorithms
│   ├── random-forest/            # Random forest ensemble
│   ├── support_vector_machine/   # SVM examples
│   ├── k-fold/                   # Cross-validation examples
│   ├── k-fold_for_iris-data-set/ # K-fold with Iris dataset
│   ├── test_and_train_data_Set/  # Train/test splitting
│   ├── save-linear-reg-model/    # Model persistence
│   ├── load_digit_data_set/      # Digit recognition dataset
│   └── Employee_retention_modal/ # Employee retention prediction
├── unsuperwised/                  # Unsupervised Learning Examples
│   ├── k mean clustering/        # K-means clustering
│   ├── naive-based-spam-detection/  # Spam detection with Naive Bayes
│   └── naive-based-titatic-usecase/  # Titanic survival prediction
├── L1&L2Regulisation/            # Regularization techniques
├── hyper-params-tunning/         # Hyperparameter optimization
├── insamble-learning/            # Ensemble methods
├── outlier-removal-IQR/          # Outlier detection and removal
├── pca/                          # Principal Component Analysis
├── knn/                          # K-Nearest Neighbors
└── README.md
```

## 🎓 Supervised Learning

### 📈 Regression Examples

#### Linear Regression (`superwised/linear-reg/`)
- **Single Variable**: Home price prediction based on area
- **Time Series**: Canada per capita income analysis (1970-2016)
- **Dataset**: `homeprices.csv`, `canada_per_capita_income.csv`
- **Key Concepts**: Simple linear regression, trend analysis

#### Multiple Linear Regression (`superwised/linear-reg-multivalue/`)
- **Multi-Variable**: Home price prediction with area, bedrooms, age
- **Data Preprocessing**: Missing value handling, data imputation
- **Advanced Features**: Feature selection, correlation analysis

#### Categorical Data Preprocessing (`superwised/dummy_variable_and_one_hot_encoding/`)
- **Modern Approach**: Scikit-learn `OneHotEncoder` with `ColumnTransformer`
- **Complete Pipeline**: Data preprocessing, model training, predictions
- **Practical Examples**:
  - **Standalone Script**: `prediction_example.py` - Complete workflow demonstration
  - **Jupyter Notebook**: Step-by-step categorical data handling
  - **Prediction Guide**: `README_PREDICTIONS.md` - Comprehensive documentation
- **Key Features**:
  - Production-ready preprocessing pipelines
  - Multiple prediction scenarios (single, batch, comparison)
  - Reusable prediction functions
  - Model interpretation and coefficient analysis
  - Error handling and best practices

### 🎯 Classification Examples

#### Logistic Regression (`superwised/logistic_regression/`)
- **Binary Classification**: Insurance purchase prediction
- **Dataset**: `insurance_data.csv`
- **Key Concepts**: Sigmoid function, probability prediction, decision boundaries

#### Multi-Class Logistic Regression (`superwised/logistic_regression-multivalue/`)
- **Multi-Class Classification**: Extended classification problems
- **Advanced Features**: One-vs-rest, multinomial classification

#### Decision Trees (`superwised/decesion_tree/`)
- **Tree-Based Learning**: Decision tree algorithms
- **Interpretability**: Visual decision paths, feature importance

#### Random Forest (`superwised/random-forest/`)
- **Ensemble Method**: Multiple decision trees
- **Improved Accuracy**: Reduced overfitting, better generalization

#### Support Vector Machine (`superwised/support_vector_machine/`)
- **SVM Classification**: Linear and non-linear classification
- **Kernel Methods**: RBF, polynomial kernels

#### K-Nearest Neighbors (`knn/`)
- **Instance-Based Learning**: Lazy learning algorithm
- **Distance Metrics**: Euclidean, Manhattan distance

### 🔄 Model Evaluation & Validation

#### Cross-Validation (`superwised/k-fold/`)
- **K-Fold Validation**: Model performance evaluation
- **Bias-Variance Tradeoff**: Understanding model generalization

#### Iris Dataset Example (`superwised/k-fold_for_iris-data-set/`)
- **Classic Dataset**: Iris flower classification
- **Complete Pipeline**: Data loading, preprocessing, validation

#### Train/Test Splitting (`superwised/test_and_train_data_Set/`)
- **Data Splitting**: Proper train/test separation
- **Performance Metrics**: Accuracy, precision, recall, F1-score

### 💾 Model Persistence (`superwised/save-linear-reg-model/`)
- **Joblib Method**: Efficient model serialization
- **Pickle Method**: Standard Python serialization
- **Model Deployment**: Loading and using saved models

### 🏢 Real-World Applications

#### Employee Retention (`superwised/Employee_retention_modal/`)
- **HR Analytics**: Predicting employee turnover
- **Business Impact**: Data-driven HR decisions

#### Digit Recognition (`superwised/load_digit_data_set/`)
- **Computer Vision**: Handwritten digit classification
- **Feature Engineering**: Image data preprocessing

## 🔍 Unsupervised Learning

### 🎯 Clustering

#### K-Means Clustering (`unsuperwised/k mean clustering/`)
- **Centroid-Based Clustering**: Grouping similar data points
- **Applications**: Customer segmentation, data exploration
- **Key Concepts**: Elbow method, cluster validation

### 📊 Probabilistic Models

#### Naive Bayes - Spam Detection (`unsuperwised/naive-based-spam-detection/`)
- **Text Classification**: Email spam detection
- **Probabilistic Learning**: Bayes theorem application
- **Feature Engineering**: Text preprocessing, TF-IDF

#### Naive Bayes - Titanic Survival (`unsuperwised/naive-based-titatic-usecase/`)
- **Survival Prediction**: Titanic passenger survival analysis
- **Historical Dataset**: Real-world classification problem
- **Feature Analysis**: Age, class, gender impact on survival

## 🚀 Advanced Topics

### 🎛 Regularization Techniques (`L1&L2Regulisation/`)
- **L1 Regularization (Lasso)**: Feature selection, sparsity
- **L2 Regularization (Ridge)**: Preventing overfitting
- **Elastic Net**: Combining L1 and L2 regularization
- **Dataset**: Melbourne housing data
- **Applications**: High-dimensional data, feature selection

### ⚙️ Hyperparameter Tuning (`hyper-params-tunning/`)
- **Grid Search**: Systematic parameter optimization
- **Random Search**: Efficient parameter exploration
- **Cross-Validation**: Robust parameter selection
- **Model Selection**: Finding the best model configuration

### 🎭 Ensemble Learning (`insamble-learning/`)
- **Bagging**: Bootstrap aggregating
- **Boosting**: Sequential model improvement
- **Voting Classifiers**: Combining multiple algorithms
- **Stacking**: Meta-learning approaches
- **Dataset**: Diabetes prediction

### 🎯 Dimensionality Reduction (`pca/`)
- **Principal Component Analysis**: Feature space reduction
- **Variance Preservation**: Maintaining data information
- **Visualization**: High-dimensional data plotting
- **Applications**: Data compression, noise reduction

### 🔍 Outlier Detection (`outlier-removal-IQR/`)
- **IQR Method**: Interquartile range outlier detection
- **Statistical Approach**: Robust outlier identification
- **Data Cleaning**: Improving model performance
- **Dataset**: Heights data analysis

## 🛠 Technologies Used

### Core Libraries
- **Python 3.x** - Programming language
- **Jupyter Notebook** - Interactive development environment
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **matplotlib** - Data visualization and plotting

### Machine Learning
- **scikit-learn** - Comprehensive ML library
  - `LinearRegression`, `LogisticRegression` - Regression and classification
  - `DecisionTreeClassifier`, `RandomForestClassifier` - Tree-based algorithms
  - `SVC`, `SVR` - Support Vector Machines
  - `KNeighborsClassifier` - K-Nearest Neighbors
  - `KMeans` - Clustering algorithms
  - `PCA` - Dimensionality reduction
  - `OneHotEncoder`, `StandardScaler` - Data preprocessing
  - `ColumnTransformer` - Feature preprocessing pipelines
  - `GridSearchCV` - Hyperparameter tuning
  - `cross_val_score` - Model validation

### Model Persistence & Deployment
- **joblib** - Efficient model serialization
- **pickle** - Standard Python object serialization

### Specialized Libraries
- **seaborn** - Statistical data visualization
- **plotly** - Interactive plotting
- **scipy** - Scientific computing

## 🏁 Getting Started

### Prerequisites

Make sure you have Python 3.x installed along with the required packages:

```bash
# Essential packages
pip install pandas numpy scikit-learn matplotlib jupyter

# Additional packages for advanced examples
pip install seaborn plotly scipy joblib
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

## ⚡ Quick Start Examples

### 🚀 Best Starting Points

#### 1. Complete Prediction Pipeline (Recommended)
```bash
cd superwised/dummy_variable_and_one_hot_encoding
python prediction_example.py
```
**What you'll see**: Complete ML workflow from data loading to predictions with professional output

#### 2. Basic Linear Regression
```bash
# Navigate to basic examples
cd superwised/linear-reg/linear_reg_home
# Open the Jupyter notebook
jupyter notebook linear-reg_home_area.ipynb
```

#### 3. Classification Example
```bash
cd superwised/logistic_regression
jupyter notebook index.ipynb
```

#### 4. Clustering Example
```bash
cd unsuperwised/k\ mean\ clustering
jupyter notebook
```

### 📊 Expected Output (Prediction Example)
```
🏠 House Price Prediction with Categorical Data
==================================================
Dataset Overview:
Shape: (13, 3)
Columns: ['town', 'area', 'price']

🔮 Making Predictions
Example 1: Single Prediction
   Input: Monroe Township, 2800 sq ft
   Predicted Price: $612,500.00

Example 2: Multiple Predictions
   1. Monroe Township, 2500 sq ft → $587,500.00
   2. West Windsor, 3500 sq ft → $675,000.00
   3. Robinsville, 2800 sq ft → $598,750.00
```

### 🎯 Key Learning Concepts

#### ✅ **Consistent Preprocessing**
- Use the same preprocessor for training and prediction
- Transform new data with `preprocessor.transform()` (not `fit_transform()`)

#### ✅ **Proper Data Format**
- Create DataFrames with identical column structure
- Maintain feature order and naming consistency

#### ✅ **Multiple Scenarios**
- Single predictions for individual cases
- Batch predictions for multiple inputs
- Comparison analysis across different categories

#### ✅ **Production-Ready Pipelines**
- Use `ColumnTransformer` for mixed data types
- Implement proper error handling and validation
- Create reusable prediction functions

## 📊 Featured Datasets

### 🏠 Real Estate Data
- **Home Prices**: Area, bedrooms, age, town, price
- **Applications**: Regression, categorical encoding, feature engineering
- **Complexity Levels**: Single variable → Multiple variables → Categorical features

### 📈 Financial & Economic Data
- **Canada Income**: Historical per capita income (1970-2016)
- **Insurance**: Customer demographics and purchase decisions
- **Applications**: Time series analysis, classification

### 🚗 Automotive Data
- **Car Prices**: Model, mileage, age, price
- **Applications**: Categorical encoding, price prediction

### 🏥 Healthcare Data
- **Diabetes**: Patient health metrics and diagnosis
- **Applications**: Classification, ensemble learning

### 👥 HR & Business Data
- **Employee Retention**: Job satisfaction, performance, turnover
- **Titanic Survival**: Passenger demographics and survival
- **Applications**: Business analytics, historical analysis

### 🔢 Classic ML Datasets
- **Iris Flowers**: Sepal/petal measurements, species classification
- **Digit Recognition**: Handwritten digit images
- **Heights**: Statistical distribution analysis
- **Applications**: Benchmarking, algorithm comparison

## 🎓 Learning Outcomes

### 📚 Fundamental Concepts
- **Supervised vs Unsupervised Learning**: Understanding different ML paradigms
- **Regression vs Classification**: Predicting continuous vs categorical outcomes
- **Model Evaluation**: Cross-validation, train/test splits, performance metrics
- **Overfitting & Underfitting**: Bias-variance tradeoff, regularization

### 🔧 Technical Skills
- **Data Preprocessing**: Cleaning, encoding, scaling, feature engineering
- **Algorithm Implementation**: Linear/logistic regression, trees, SVM, clustering
- **Pipeline Development**: End-to-end ML workflows, preprocessing pipelines
- **Model Deployment**: Saving, loading, and using trained models

### 📊 Data Handling
- **Mixed Data Types**: Numerical, categorical, text data preprocessing
- **Missing Values**: Imputation strategies, data quality assessment
- **Feature Engineering**: Creating, selecting, and transforming features
- **Dimensionality Reduction**: PCA, feature selection techniques

### 🚀 Advanced Topics
- **Ensemble Methods**: Combining multiple models for better performance
- **Hyperparameter Tuning**: Grid search, random search, optimization
- **Regularization**: L1/L2 penalties, preventing overfitting
- **Outlier Detection**: Identifying and handling anomalous data

### 💼 Practical Applications
- **Real-World Projects**: Business problems, data-driven decision making
- **Production Pipelines**: Scalable, maintainable ML systems
- **Best Practices**: Code organization, documentation, reproducibility

## 🤝 Contributing

We welcome contributions! Here are ways you can help:

### 🆕 New Examples
- Add examples for new algorithms (Neural Networks, XGBoost, etc.)
- Create industry-specific use cases (Finance, Healthcare, Marketing)
- Implement advanced techniques (Deep Learning, NLP, Computer Vision)

### 📈 Improvements
- Enhance existing notebooks with better explanations
- Add more comprehensive error handling
- Improve data visualization and insights
- Optimize code performance and readability

### 📚 Documentation
- Expand prediction guides and tutorials
- Add troubleshooting sections
- Create video tutorials or blog posts
- Translate documentation to other languages

### 🔧 Technical Enhancements
- Add automated testing for notebooks
- Create Docker containers for easy setup
- Implement CI/CD pipelines
- Add interactive web demos

### 📊 Datasets
- Contribute new, interesting datasets
- Clean and document existing datasets
- Add data validation and quality checks

## 🌟 Acknowledgments

- **scikit-learn community** for excellent documentation and examples
- **Kaggle** for providing diverse datasets
- **Jupyter Project** for the interactive notebook environment
- **Open source contributors** who make machine learning accessible

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

**Happy Learning! 🚀** Start with the prediction example and explore the fascinating world of machine learning!