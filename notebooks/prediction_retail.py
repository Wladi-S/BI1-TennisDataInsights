from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from time import time

X = train_df
y = train_df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# spalten zum droppen
drop_colmn = ['Sales', 'Customers', 'Date']

# für StandardScaler
ss_colmn = ['Year', 'Month', 'Day', 'WeekOfYear', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']

# für One Hot Encoding
ohe_colmn = ['StoreType', 'Assortment', 'PromoInterval']


# Vorverarbeitungspipeline erstellen
preprocessor = ColumnTransformer(
    transformers=[
        ('drop_cols', 'drop', drop_colmn),
        ('scale', StandardScaler(), ss_colmn),
        ('onehot', OneHotEncoder(handle_unknown='ignore'), ohe_colmn)
    ], remainder='passthrough')

# Modellpipeline erstellen
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_jobs=-1, random_state=42))
])

# Hyperparameter-Raster definieren
param_grid = {
    'regressor__n_estimators': [300], # [100, 200, 300],
    'regressor__max_depth': [40], # [20, 30, 40],
    #'regressor__max_features': [None, 'sqrt', 'log2'],
    'regressor__min_samples_split': [5], #[5, 10, 15],
}

# Grid-Suche durchführen
tscv = TimeSeriesSplit(n_splits=5)
gs = GridSearchCV(model, param_grid=param_grid, cv=tscv, n_jobs=-1)
gs.fit(X_train, y_train)

y_pred_random_forest = gs.predict(X_test)

print(f"Beste Hyperparameter-Kombination: {gs.best_params_}")
print(f"Accuracy Trainingsdaten: {gs.score(X_train, y_train)}")
print(f"Accuracy Testdaten: {gs.score(X_test, y_test)}")