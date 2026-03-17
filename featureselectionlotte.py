#%% 
import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split, StratifiedKFold, GridSearchCV, cross_val_score)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier

#%% 
# Data inladen 

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

#%% Feautre selectie 
# Dit voorkomt overfitting, en maakt het model eenvoudiger en sneller.
# Er zitten namelijk 9000 features in en 837 patienten (zie geprinte info)
#%% PCA - Principle component analysis 

#from sklearn.decomposition import PCA

# aantal componenten kiezen dat behouden moet worden 
pca = PCA(n_components = 0.95) 
#toepassen ( fitten en transformen op train data)
X_pca_train = pca.fit_transform(X_train_scaled)

# alleen transformen op test data 
X_pca_test = pca.transform(X_test_scaled) 

print(f"Aantal features voor PCA: {X_train_scaled.shape[1]}")
print(f"Aantal componenten na PCA (95% variantie): {X_pca_train.shape[1]}")


#%%pipeline maken voor PCA

# vaste instellingen 
fixed_scaler = RobustScaler() 

#pipeline maken
pca_par_pipe = Pipeline([('scaler', fixed_scaler), 
                         ('pca', PCA()), 
                         ('clf', SVC(kernel = 'rbf', probability=True, class_weight='balanced'))])

#te testen parameters definieren 
# n components -> nog geen onderbouwing 
# whiten -> 
# svd solver andere kunnen niet met n_components waardes
param_grid = {'pca__n_components' : [ 0.80, 0.90, 0.95], 
              'pca__whiten' : [True, False], 
              'pca__svd_solver' : ['auto']} #hoeveel componenten we behouden 


# beste combinatie zoeken 
grid = GridSearchCV(pca_par_pipe, 
                    param_grid, 
                    cv=5, 
                    scoring='roc_auc',
                    n_jobs=1)
grid.fit(X_train, y_train)

print(f"Beste parameters: {grid.best_params_}")
print(f"Beste Cross-Validatie AUC: {grid.best_score_:4f}")

#%%
# %% Lasso 
# juiste penalty en juiste Ltype kiezen 
# solver is de 'rekenmachine' die de juiste gewichten voor je features berekend. liblinear is specifiek voor kleine datasets. 
# importeren wanneer nodig 
lasso_select = SelectFromModel(LogisticRegression(penalty='l1', solver = 'liblinear', C=1.0 , class_weight= 'balanced', random_state= 42))

lasso_select.fit(X_train_scaled, y_train)

X_lasso_train = lasso_select.transform(X_train_scaled)
X_lasso_test = lasso_select.transform(X_test_scaled)

n_features_lasso = X_lasso_train.shape[1]
print(f"Aantal features voor Lasso: {X_train_scaled.shape[1]}")
print(f"Lasso heeft {n_features_lasso} over")

# %% pipeline maken voor lasso

# vaste instellingen 
fixed_scaler = RobustScaler() 

#pipeline maken
lasso_par_pipe = Pipeline([
    ('scaler', RobustScaler()), 
    # We gebruiken LogisticRegression met L1 als de 'motor' voor selectie
    ('selector', SelectFromModel(LogisticRegression(penalty='l1', class_weight='balanced'))),
    ('clf', SVC(kernel='rbf', probability=True, class_weight='balanced'))
])

#te testen parameters definieren 
# n components -> nog geen onderbouwing 
# whiten -> 
# svd solver andere kunnen niet met n_components waardes
param_grid = {'selector__estimator__solver' : ['liblinear'], #rekenkundige methode, enige mogelijkheden met L1
              'selector__estimator__random_state' : [ 42 ],  # gekozen getal maakt eigenlijk niet uit, gaat erom dat ie overal hetzelfde is, 
              'selector__estimator__C' : [0.01, 0.1, 1.0, 10.0]} 


# beste combinatie zoeken 
grid = GridSearchCV(lasso_par_pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=1 )
grid.fit(X_train, y_train) 

print(f"Beste parameters: {grid.best_params_}")
print(f"Beste Cross-Validatie AUC: {grid.best_score_:4f}")

#%% 
#%% pipeline PCA feature selectie op clf KNN

# vaste instellingen 
fixed_scaler = RobustScaler() 
fixed_classifier = KNeighborsClassifier(n_neighbors=3)

#pipeline maken
pca_par_pipe = Pipeline([('scaler', fixed_scaler), 
                         ('pca', PCA()), 
                         ('clf', fixed_classifier)])

#te testen parameters definieren 
# n components -> nog geen onderbouwing 
# whiten -> 
# svd solver andere kunnen niet met n_components waardes
param_grid = {'pca__n_components' : [ 0.80, 0.90, 0.95], 
              'pca__whiten' : [True, False], 
              'pca__svd_solver' : ['auto']} #hoeveel componenten we behouden 


# beste combinatie zoeken 
grid = GridSearchCV(pca_par_pipe, 
                    param_grid, 
                    cv=5, 
                    scoring='roc_auc',
                    n_jobs=1)
grid.fit(X_train, y_train)

print(f"Beste parameters: {grid.best_params_}")
print(f"Beste Cross-Validatie AUC: {grid.best_score_:4f}")

#%% pipeline PCA feature selectie op clf randomforest

# vaste instellingen 
fixed_scaler = RobustScaler() 
fixed_classifier = RandomForestClassifier (n_estimators = 100, random_state = 42)

#pipeline maken
pca_par_pipe = Pipeline([('scaler', fixed_scaler), 
                         ('pca', PCA()), 
                         ('clf', fixed_classifier)])

#te testen parameters definieren 
# n components -> nog geen onderbouwing 
# whiten -> 
# svd solver andere kunnen niet met n_components waardes
param_grid = {'pca__n_components' : [ 0.80, 0.90, 0.95], 
              'pca__whiten' : [True, False], 
              'pca__svd_solver' : ['auto']} #hoeveel componenten we behouden 


# beste combinatie zoeken 
grid = GridSearchCV(pca_par_pipe, 
                    param_grid, 
                    cv=5, 
                    scoring='roc_auc',
                    n_jobs=1)
grid.fit(X_train, y_train)

print(f"Beste parameters: {grid.best_params_}")
print(f"Beste Cross-Validatie AUC: {grid.best_score_:4f}")

#%% pipeline PCA feature selectie op clf XGBoost


# vaste instellingen 
fixed_scaler = RobustScaler() 
fixed_classifier = xgb.XGBClassifier(
    n_estimators= 100, 
    learning_rate = 0.1, 
    max_depth = 5, 
    use_label_encoder = False, 
    eval_metric= 'logloss'
)

#pipeline maken
pca_par_pipe = Pipeline([('scaler', fixed_scaler), 
                         ('pca', PCA()), 
                         ('clf', fixed_classifier)])

#te testen parameters definieren 
# n components -> nog geen onderbouwing 
# whiten -> 
# svd solver andere kunnen niet met n_components waardes
param_grid = {'pca__n_components' : [ 0.80, 0.90, 0.95], 
              'pca__whiten' : [True, False], 
              'pca__svd_solver' : ['auto']} #hoeveel componenten we behouden 


# beste combinatie zoeken 
grid = GridSearchCV(pca_par_pipe, 
                    param_grid, 
                    cv=5, 
                    scoring='roc_auc',
                    n_jobs=1)
grid.fit(X_train, y_train)

print(f"Beste parameters: {grid.best_params_}")
print(f"Beste Cross-Validatie AUC: {grid.best_score_:4f}")

#%%

# %% pipeline Lasso feature selection op KNN
# vaste instellingen 
fixed_scaler = RobustScaler() 
fixed_classifier = KNeighborsClassifier(n_neighbors=3)

#pipeline maken
lasso_par_pipe = Pipeline([
    ('scaler', RobustScaler()), 
    # We gebruiken LogisticRegression met L1 als de 'motor' voor selectie
    ('selector', SelectFromModel(LogisticRegression(penalty='l1', class_weight='balanced'))),
    ('clf', fixed_classifier)
])

#te testen parameters definieren 
# n components -> nog geen onderbouwing 
# whiten -> 
# svd solver andere kunnen niet met n_components waardes
param_grid = {'selector__estimator__solver' : ['liblinear'], #rekenkundige methode, enige mogelijkheden met L1
              'selector__estimator__random_state' : [ 42 ],  # gekozen getal maakt eigenlijk niet uit, gaat erom dat ie overal hetzelfde is, 
              'selector__estimator__C' : [0.01, 0.1, 1.0, 10.0]} 


# beste combinatie zoeken 
grid = GridSearchCV(lasso_par_pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=1 )
grid.fit(X_train, y_train) 

print(f"Beste parameters: {grid.best_params_}")
print(f"Beste Cross-Validatie AUC: {grid.best_score_:4f}")

#%% pipeline Lasso feature selection op random forest

# vaste instellingen 
fixed_scaler = RobustScaler() 
fixed_classifier = RandomForestClassifier (n_estimators = 100, random_state = 42)

#pipeline maken
lasso_par_pipe = Pipeline([
    ('scaler', RobustScaler()), # We gebruiken LogisticRegression met L1 als de 'motor' voor selectie
    ('selector', SelectFromModel(LogisticRegression(penalty='l1', class_weight='balanced'))),
    ('clf', fixed_classifier)])

#te testen parameters definieren 
# n components -> nog geen onderbouwing 
# whiten -> 
# svd solver andere kunnen niet met n_components waardes
param_grid = {'selector__estimator__solver' : ['liblinear'], #rekenkundige methode, enige mogelijkheden met L1
              'selector__estimator__random_state' : [ 42 ],  # gekozen getal maakt eigenlijk niet uit, gaat erom dat ie overal hetzelfde is, 
              'selector__estimator__C' : [0.01, 0.1, 1.0, 10.0]} 


# beste combinatie zoeken 
grid = GridSearchCV(lasso_par_pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=1 )
grid.fit(X_train, y_train) 

print(f"Beste parameters: {grid.best_params_}")
print(f"Beste Cross-Validatie AUC: {grid.best_score_:4f}")

#%% pipeline Lasso feature selection op XGBoost
# vaste instellingen 
fixed_scaler = RobustScaler() 
fixed_classifier = xgb.XGBClassifier(
    n_estimators= 100, 
    learning_rate = 0.1, 
    max_depth = 5, 
    use_label_encoder = False, 
    eval_metric= 'logloss'
)

#pipeline maken
lasso_par_pipe = Pipeline([
    ('scaler', RobustScaler()), # We gebruiken LogisticRegression met L1 als de 'motor' voor selectie
    ('selector', SelectFromModel(LogisticRegression(penalty='l1', class_weight='balanced'))),
    ('clf', fixed_classifier)])

#te testen parameters definieren 
# n components -> nog geen onderbouwing 
# whiten -> 
# svd solver andere kunnen niet met n_components waardes
param_grid = {'selector__estimator__solver' : ['liblinear'], #rekenkundige methode, enige mogelijkheden met L1
              'selector__estimator__random_state' : [ 42 ],  # gekozen getal maakt eigenlijk niet uit, gaat erom dat ie overal hetzelfde is, 
              'selector__estimator__C' : [0.01, 0.1, 1.0, 10.0]} 


# beste combinatie zoeken 
grid = GridSearchCV(lasso_par_pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=1 )
grid.fit(X_train, y_train) 

print(f"Beste parameters: {grid.best_params_}")
print(f"Beste Cross-Validatie AUC: {grid.best_score_:4f}")