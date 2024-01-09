import lightgbm as lgb
from data_pipeline import data_transform_pipeline
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error


data = pd.read_csv('data/train.csv')

train_data = data[data['date'] < '2017-08-01']
test_data = data[data['date'] >= '2017-08-01']

X_train = train_data
y_train = X_train.pop('sales')

X_test = test_data
y_test = X_test.pop('sales')

X_train = data_transform_pipeline.fit_transform(X_train)
X_test = data_transform_pipeline.transform(X_test)

cv_split = TimeSeriesSplit(n_splits=4, test_size=100)
model = lgb.LGBMRegressor()
parameters = {
    "max_depth": [3, 4, 5, 10],
    "num_leaves": [10, 20, 40, 100],
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "n_estimators": [50, 100, 300, 500, 700, 900],
    "colsample_bytree": [0.3, 0.5, 0.7, 1]
}


grid_search = GridSearchCV(estimator=model, cv=cv_split, param_grid=parameters)
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

prediction = grid_search.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, prediction)}")
print(f"ROOT of MSE: {mean_squared_error(y_test, prediction)**0.5}")

