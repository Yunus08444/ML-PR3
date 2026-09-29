import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

np.random.seed(42)
data = pd.DataFrame({
    'PassengerId': range(1, 101),
    'Age': [22, 38, 26, 35, np.nan, 54, 2, 27, 14, 4] + list(np.random.randint(18, 70, 90)),
    'Fare': [7.25, 71.28, 7.92, 53.10, 8.05, np.nan, 21.07, 11.13, 30.07, 16.70] + list(np.random.exponential(50, 90)),
    'Sex': ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'male', 'female', 'female'] + list(np.random.choice(['male', 'female'], 90)),
    'Embarked': ['S', 'C', 'S', 'S', 'S', 'Q', 'S', 'S', 'S', np.nan] + list(np.random.choice(['S', 'C', 'Q'], 90)),
    'Pclass': np.random.choice([1, 2, 3], 100),
    'Target': np.random.choice([0, 1], 100)
})

X = data.drop(columns=['PassengerId', 'Target'])
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_median_features = ['Age', 'Fare']              
cat_missing_category_features = ['Embarked']      
cat_mode_features = ['Sex', 'Pclass']              

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_missing_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

cat_mode_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_median_features),
        ('cat_miss', cat_missing_transformer, cat_missing_category_features),
        ('cat_mode', cat_mode_transformer, cat_mode_features)
    ],
    remainder='drop'
)

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])


X_train_prepared = full_pipeline.fit_transform(X_train)

X_test_prepared = full_pipeline.transform(X_test)

print("--- Результаты трансформации ---")
print(f"Размер исходного X_train: {X_train.shape}")
print(f"Размер обработанного X_train: {X_train_prepared.shape}")
print(f"Размер исходного X_test: {X_test.shape}")
print(f"Размер обработанного X_test: {X_test_prepared.shape}")

assert not np.isnan(X_train_prepared).any(), "Ошибка: в X_train остались NaN!"
assert not np.isnan(X_test_prepared).any(), "Ошибка: в X_test остались NaN!"

print("\n Успешно: Пайплайн применен, пропуски заполнены, утечка данных отсутствует!")