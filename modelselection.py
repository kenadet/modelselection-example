import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error

# 1. Read dataset and create dependent and independent variables
df = pd.read_csv("housing")
X = df.drop(columns= 'price') # Create independent variables, otherwise called predictors or features
y = df['price'] # Create dependent variable

# 2. Perform Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Create Proprocessing pipelines
num_features == df.select_dtypes(include=['int64', 'float64']).columns # ['age', 'size']
cat_features =  df.select_dtypes(include=['object', 'category']).columns # ['region']

cat_pipeline = Pipeline([
  ('imputer', SimpleImputer(strategy = 'most_frequent')),
  ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

num_pipeline = Pipeline([   # If there are outliers detected during describe, and EDA, we would need to remove them first before doing this
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
  ('num', num_pipeline, num_features), 
  ('cat',cat_pipeline, cat_features )
])

# 4. Extend pipeline with preprocessor with Ridge Regression 
ridge_pipeline = Pipeline([
  ('processing', preprocessor),
  ('regressor', Ridge())
])

# 5. Perform GridSearchCV with 5-fold cross-validation to tune Ridge alpha over
# given param_grid, selecting the model with the lowest MSE (using neg_mean_squared_error) on fitting.

param_grid = {'regressor__alpha': [0.01, 0.1, 1, 10, 100]}

grid_search = GridSearchCV(
    ridge_pipeline,
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error'
)

...
