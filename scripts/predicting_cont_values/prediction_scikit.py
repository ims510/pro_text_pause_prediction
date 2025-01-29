from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import complete_data_processing as dp
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# df = dp.main()
df = pd.read_pickle("/Users/madalina/Documents/M1TAL/stage_GC/pro_text/expanded_data_decale.pkl")
# Separate features and target
X = df.drop(columns=['pauseDur'])
y = df['pauseDur']
# print(X.head())
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

# Initialize and train the model
lr_model = LinearRegression()
dt_model = DecisionTreeRegressor(random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

lr_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

dt_y_pred = dt_model.predict(X_test)
lr_y_pred = lr_model.predict(X_test)
rf_y_pred = rf_model.predict(X_test)

dt_r2 = r2_score(y_test, dt_y_pred)
lr_r2 = r2_score(y_test, lr_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)

dt_mse = mean_squared_error(y_test, dt_y_pred)
lr_mse = mean_squared_error(y_test, lr_y_pred)
rf_mse = mean_squared_error(y_test, rf_y_pred)
# Evaluate the model
# lr_score = lr_model.score(X_test, y_test)
print(f'R^2 Score for linear regression: {lr_r2:.4f}')
print(f'Mean Squared Error for linear regression: {lr_mse:.4f}')
print(f"R^2 Score for decision tree: {dt_r2:.4f}")
print(f"Mean Squared Error for decision tree: {dt_mse:.4f}")
print(f"R^2 Score for random forest: {rf_r2:.4f}")
print(f"Mean Squared Error for random forest: {rf_mse:.4f}")


def evaluate_model_with_cv(model, X, y, cv=5):
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    mse_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    print(f"Cross-validated R² Score: {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
    print(f"Cross-validated Mean Squared Error: {-mse_scores.mean():.4f} ± {mse_scores.std():.4f}")

print("Linear Regression:")
evaluate_model_with_cv(lr_model, X_train, y_train)

print("\nDecision Tree Regressor:")
evaluate_model_with_cv(dt_model, X_train, y_train)

print("\nRandom Forest Regressor:")
evaluate_model_with_cv(rf_model, X_train, y_train)

# Example for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Train the model with grid search
grid_search.fit(X_train, y_train)

# Best parameters and model
best_params = grid_search.best_params_
best_rf_model = grid_search.best_estimator_

print(f"Best Parameters: {best_params}")
# Predicting with the best random forest model
y_best_pred = best_rf_model.predict(X_test)

# Evaluating the best model
r2_best_rf = r2_score(y_test, y_best_pred)
mse_best_rf = mean_squared_error(y_test, y_best_pred)

print(f"Best R² Score (Random Forest): {r2_best_rf:.4f}")
print(f"Best Mean Squared Error (Random Forest): {mse_best_rf:.4f}")

# Cross-validate the best model
print("\nBest Random Forest Regressor (Cross-validated):")
evaluate_model_with_cv(best_rf_model, X_train, y_train)
