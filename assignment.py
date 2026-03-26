
#%% ==================================================================================
# Basis
# ====================================================================================
%pip install xgboost
#%% Data loading functions

from numpy import linspace
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.model_selection import (train_test_split, StratifiedKFold, GridSearchCV, cross_val_score)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import (classification_report, roc_auc_score, average_precision_score,confusion_matrix, ConfusionMatrixDisplay)
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

#%% Loading data

def load_data():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    # Removed the intermediate 'ecg_data' folder from the path
    data = pd.read_csv(os.path.join(this_directory, 'ecg_data.csv'), index_col=0)
    return data

data = load_data()

#%% Kijken hoe de dataset is opgebouwd 

print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')
print(data.head())

print("\nMissing values per column:")
print(data.isnull().sum())

print("\nTotal missing values:")
print(data.isnull().sum().sum())

print("\nColumn names:")
print(data.columns)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print("\nUnieke klassen:")
print(y.unique())

print(y.value_counts())

#%% Data opsplitsen in train en test

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Zorgt ervoor dat verdeling 0 en 1 gelijk blijft, want we hadden niet goed verdeelde data 
)

#%% ==================================================================================
# Feature selection and k selection
# ====================================================================================

# Dit deel is nog met cross validatie ipv nested-cross validatie. Ik heb dit niet super goed uitgewerkt ofzo. Dus niet zomaar overnemen.

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # Cross-validation instellen

k_values = [20, 50, 100, 150, 200, 300, 500, 1000]
k_scores = []

for k in k_values:
    pipe_k = Pipeline([
        ('scaler', RobustScaler()),
        ('selector', SelectKBest(score_func=f_classif, k=k)),
        ('clf', SVC(kernel='rbf', gamma='scale', C=1, probability=True, class_weight='balanced'))
    ])

    scores = cross_val_score(pipe_k, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    mean_score = scores.mean()
    k_scores.append(mean_score)
    print(f"k = {k:4d} | mean CV ROC-AUC = {mean_score:.4f}")

best_k = k_values[int(np.argmax(k_scores))]
print(f"\nBeste k op basis van CV ROC-AUC: {best_k}")

<<<<<<< HEAD
# Plot k-analyse
plt.figure(figsize=(8, 5))
plt.plot(k_values, k_scores, marker='o')
plt.xlabel("Aantal geselecteerde features (k)")
plt.ylabel("Mean CV ROC-AUC")
plt.title("Hyperparameter analyse: SelectKBest k")
plt.grid(True)
plt.tight_layout()
plt.show()

#%% ==================================================================================
# SVM RBF
# ====================================================================================

# Pipeline maken
fixed_scaler = RobustScaler() 
fixed_k = best_k    # Hiervoor gevonden

svm_rbf_pipe = Pipeline([
    ('scaler', fixed_scaler),
    ('selector', SelectKBest(score_func=f_classif, k=fixed_k)),
    ('clf', SVC(kernel='rbf', probability=True))
])

# Hyperparameter grid
svm_rbf_param_grid = {
    'clf__C': [1e-2, 1e-1, 1, 10, 100],
    'clf__gamma': [1e-4, 1e-3, 1e-2, 1e-1, 1]
}

# Inner en outer CV definiëren
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Inner grid search
svm_rbf_grid = GridSearchCV(
    estimator=svm_rbf_pipe,
    param_grid=svm_rbf_param_grid,
    scoring='roc_auc',
    cv=inner_cv,
=======
#%%

#%% ==================================================================================
# SVM inclusief hyperparameters
# ====================================================================================

#%% Extra imports
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

#%% Pipeline maken
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(score_func=f_classif)),
    ('svm', SVC(kernel='poly', gamma='scale', probability=True))
])

#%% Hyperparameter grid
param_grid = {
    'selector__k': [20, 50, 100, 150, 200, 300, 500, 1000],
    'svm__degree': [1, 3, 5],
    'svm__coef0': [0.01, 0.5, 1],
    'svm__C': [0.01, 0.5, 1]
}

#%% Cross-validation instellen
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#%% Grid search
grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv,
>>>>>>> 7c902b05d972c0319aacfd70519cd065c535a270
    n_jobs=-1,
    verbose=1
)

<<<<<<< HEAD
# Nested CV uitvoeren
svm_rbf_nested_scores = cross_validate(
    estimator=svm_rbf_grid,
    X=X_train,
    y=y_train,
    cv=outer_cv,
    scoring='roc_auc',
    n_jobs=-1,
    return_train_score=False,
    return_estimator=True
)

# svm_rbf_grid.fit(X_train, y_train)

# print(f"\nBest params for SVM-RBF:")
# print(svm_rbf_grid.best_params_)
# print(f"Best CV ROC-AUC: {svm_rbf_grid.best_score_:.4f}")

# Resultaten
print("Outer fold ROC-AUC scores:")
print(svm_rbf_nested_scores['test_score'])

print(f"\nGemiddelde nested ROC-AUC: {svm_rbf_nested_scores['test_score'].mean():.4f}")
print(f"Standaarddeviatie nested ROC-AUC: {svm_rbf_nested_scores['test_score'].std():.4f}")

print("\nBeste parameters per outer fold:")
for i, est in enumerate(svm_rbf_nested_scores['estimator'], 1):
    print(f"Fold {i}: {est.best_params_}")

#%% ==================================================================================
# Final model SVM RBF 
# ====================================================================================

# Final search op volledige trainingsset
svm_rbf_grid.fit(X_train, y_train)

# Beste parameters
print("\nFinal best params for SVM-RBF:")
print(svm_rbf_grid.best_params_)

# Beste CV-score op de volledige trainingsset
print(f"Final CV ROC-AUC on full training set: {svm_rbf_grid.best_score_:.4f}")

# Definitieve model
final_svm_rbf_model = svm_rbf_grid.best_estimator_


#%% ==================================================================================
# SVM Poly
# ====================================================================================

# Pipeline maken
fixed_scaler = RobustScaler()
fixed_k = best_k

svm_poly_pipe = Pipeline([
    ('scaler', fixed_scaler),
    ('selector', SelectKBest(score_func=f_classif, k=fixed_k)),
    ('clf', SVC(kernel='poly', probability=True))
])

# Hyperparameter grid
svm_poly_param_grid = {
    'clf__C': [1e-2, 1e-1, 1, 10, 100],
    'clf__gamma': [1e-4, 1e-3, 1e-2, 1e-1, 1],
    'clf__degree': [2, 3, 4],
    'clf__coef0': [0, 0.5, 1]
}

# Inner en outer CV
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid search
svm_poly_grid = GridSearchCV(
    estimator=svm_poly_pipe,
    param_grid=svm_poly_param_grid,
    scoring='roc_auc',
    cv=inner_cv,
    n_jobs=-1,
    verbose=1
)

# svm_poly_grid.fit(X_train, y_train)

# print(f"\nBest params for SVM-RBF:")
# print(svm_poly_grid.best_params_)
# print(f"Best CV ROC-AUC: {svm_poly_grid.best_score_:.4f}")

# Nested cross-validation
svm_poly_nested_scores = cross_validate(
    estimator=svm_poly_grid,
    X=X_train,
    y=y_train,
    cv=outer_cv,
    scoring='roc_auc',
    n_jobs=-1,
    return_estimator=True,
    return_train_score=False
)

# Resultaten
print("Outer fold ROC-AUC scores:")
print(svm_poly_nested_scores['test_score'])

print(f"\nMean nested ROC-AUC: {np.mean(svm_poly_nested_scores['test_score']):.4f}")
print(f"Std nested ROC-AUC: {np.std(svm_poly_nested_scores['test_score']):.4f}")

print("\nBest params per outer fold:")
for i, est in enumerate(svm_poly_nested_scores['estimator'], 1):
    print(f"Fold {i}: {est.best_params_}")

#%% ==================================================================================
# Final model SVM Poly
# ====================================================================================

# Final search op volledige trainingsset
svm_poly_grid.fit(X_train, y_train)

# Beste parameters
print("\nFinal best params for SVM-Poly:")
print(svm_poly_grid.best_params_)

# Beste CV-score op de volledige trainingsset
print(f"Final CV ROC-AUC on full training set: {svm_poly_grid.best_score_:.4f}")

# Definitieve model
final_svm_poly_model = svm_poly_grid.best_estimator_

#%% ==================================================================================
# K Nearest Neighbours (KNN)
# ====================================================================================

# Pipeline maken
fixed_scaler = RobustScaler()
fixed_k = best_k

knn_pipe = Pipeline([
    ('scaler', fixed_scaler),
    ('selector', SelectKBest(score_func=f_classif, k=fixed_k)),
    ('clf', KNeighborsClassifier())
])

# Hyperparameter grid
knn_param_grid = {
    'clf__n_neighbors': [3,5,7,9,11,13,15,17,19],
    'clf__weights': ['uniform','distance'],
    'clf__p': [1,2],
    'clf__algorithm': ['auto','ball_tree','kd_tree'],
    'clf__leaf_size': [20,30,40]
}

# Inner en outer CV
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid search
knn_grid = GridSearchCV(
    estimator=knn_pipe,
    param_grid=knn_param_grid,
    scoring='roc_auc',
    cv=inner_cv,
    n_jobs=-1,
    verbose=1
)

# Nested cross-validation
knn_nested_scores = cross_validate(
    estimator=knn_grid,
    X=X_train,
    y=y_train,
    cv=outer_cv,
    scoring='roc_auc',
    n_jobs=-1,
    return_estimator=True,
    return_train_score=False
)

# Resultaten
print("Outer fold ROC-AUC scores:")
print(knn_nested_scores['test_score'])

print(f"\nMean nested ROC-AUC: {np.mean(knn_nested_scores['test_score']):.4f}")
print(f"Std nested ROC-AUC: {np.std(knn_nested_scores['test_score']):.4f}")

print("\nBest params per outer fold:")
for i, est in enumerate(knn_nested_scores['estimator'], 1):
    print(f"Fold {i}: {est.best_params_}")

#%% ==================================================================================
# Final model KNN
# ====================================================================================

# Final search op volledige trainingsset
knn_grid.fit(X_train, y_train)

# Beste parameters
print("\nFinal best params for KNN:")
print(knn_grid.best_params_)

# Beste CV-score op de volledige trainingsset
print(f"Final CV ROC-AUC on full training set: {knn_grid.best_score_:.4f}")

# Definitieve model
final_knn_model = knn_grid.best_estimator_

#%% ==================================================================================
# Random Forest
# ====================================================================================

# Pipeline maken
fixed_k = best_k

rf_pipe = Pipeline([
    ('selector', SelectKBest(score_func=f_classif, k=fixed_k)),
    ('clf', RandomForestClassifier(random_state=42))
])
# Bij Random Forest hoef je in principe niet te schalen, omdat bomen niet afstandsgebaseerd zijn.

# Hyperparameter grid
rf_param_grid = {
    'clf__n_estimators': [100, 200, 500],
    'clf__max_depth': [None, 5, 10, 20, 30],
    'clf__max_features': ['sqrt', 'log2', None],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4]
}

# Inner en outer CV
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid search
rf_grid = GridSearchCV(
    estimator=rf_pipe,
    param_grid=rf_param_grid,
    scoring='roc_auc',
    cv=inner_cv,
    n_jobs=-1,
    verbose=1
)

# Nested cross-validation
rf_nested_scores = cross_validate(
    estimator=rf_grid,
    X=X_train,
    y=y_train,
    cv=outer_cv,
    scoring='roc_auc',
    n_jobs=-1,
    return_estimator=True,
    return_train_score=False
)

# Resultaten
print("Outer fold ROC-AUC scores:")
print(rf_nested_scores['test_score'])

print(f"\nMean nested ROC-AUC: {np.mean(rf_nested_scores['test_score']):.4f}")
print(f"Std nested ROC-AUC: {np.std(rf_nested_scores['test_score']):.4f}")

print("\nBest params per outer fold:")
for i, est in enumerate(rf_nested_scores['estimator'], 1):
    print(f"Fold {i}: {est.best_params_}")

#%% ==================================================================================
# Final model Random Forest
# ====================================================================================

# Final search op volledige trainingsset
rf_grid.fit(X_train, y_train)

# Beste parameters
print("\nFinal best params for Random Forest:")
print(rf_grid.best_params_)

# Beste CV-score op de volledige trainingsset
print(f"Final CV ROC-AUC on full training set: {rf_grid.best_score_:.4f}")

# Definitieve model
final_rf_model = rf_grid.best_estimator_


# %% ===================================================================================
# XGBoost classifier 
# ======================================================================================
# 1. Calculate the clinical imbalance ratio dynamically for the training fold
# This ensures we penalize missed 'Abnormal' (1) cases heavily
imbalance_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# 2. Pipeline creation
fixed_k = best_k

xgb_pipe = Pipeline([
    ('selector', SelectKBest(score_func=f_classif, k=fixed_k)),
    ('clf', xgb.XGBClassifier(
        objective='binary:logistic', 
        eval_metric='auc',
        scale_pos_weight=imbalance_ratio, # Crucial for your 681 vs 146 split
        random_state=42
    ))
])

# 3. Hyperparameter grid for the Inner CV
# learning_rate prevents the sequential trees from over-correcting too aggressively
xgb_param_grid = {
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth': [3, 5, 7],
    'clf__learning_rate': [0.01, 0.05, 0.1],
    'clf__subsample': [0.8, 1.0],         # Uses a fraction of patients per tree to prevent overfitting
    'clf__colsample_bytree': [0.8, 1.0]   # Uses a fraction of features per tree
}

# Inner en outer CV
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid search (The Lab)
xgb_grid = GridSearchCV(
    estimator=xgb_pipe,
    param_grid=xgb_param_grid,
    scoring='roc_auc',
    cv=inner_cv,
    n_jobs=-1,
    verbose=1
)

# Nested cross-validation (The Clinical Trial)
xgb_nested_scores = cross_validate(
    estimator=xgb_grid,
    X=X_train,
    y=y_train,
    cv=outer_cv,
    scoring='roc_auc',
    n_jobs=-1,
    return_estimator=True,
    return_train_score=False
)

# Resultaten
print("Outer fold ROC-AUC scores (XGBoost):")
print(xgb_nested_scores['test_score'])

print(f"\nMean nested ROC-AUC: {np.mean(xgb_nested_scores['test_score']):.4f}")
print(f"Std nested ROC-AUC: {np.std(xgb_nested_scores['test_score']):.4f}")

print("\nBest params per outer fold:")
for i, est in enumerate(xgb_nested_scores['estimator'], 1):
    print(f"Fold {i}: {est.best_params_}")

#%% ==================================================================================
# Final model XGBoost
# ====================================================================================

# Final search op volledige trainingsset
xgb_grid.fit(X_train, y_train)

# Beste parameters
print("\nFinal best params for XGBoost:")
print(xgb_grid.best_params_)

# Beste CV-score op de volledige trainingsset
print(f"Final CV ROC-AUC on full training set: {xgb_grid.best_score_:.4f}")

# Definitieve model
final_xgb_model = xgb_grid.best_estimator_
=======
grid.fit(X_train, y_train)

#%% Beste instellingen
print("Beste parameters:")
print(grid.best_params_)
print(f"Beste cross-val ROC-AUC: {grid.best_score_:.4f}")

#%% Evaluatie op testset
best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\nTestset resultaten:")
print(classification_report(y_test, y_pred))
print(f"Test ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(f"Test Average Precision: {average_precision_score(y_test, y_prob):.4f}")
>>>>>>> 7c902b05d972c0319aacfd70519cd065c535a270
