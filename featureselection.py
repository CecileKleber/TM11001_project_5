# ## Data loading and cleaning
# Below are functions to load the dataset of your choice. After that, it is all up to you to create and evaluate a classification method. Beware, there may be missing values in these datasets. Good luck!


#%% Data loading functions. Uncomment the one you want to use
#from ecg.load_data import load_data
from numpy import linspace
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV



def load_data():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(this_directory, 'ecg_data.csv'), index_col=0)
    return data


data = load_data()

#%% checks doen aan de dataset


# Printen van het aantal samples en het aantal kolommen, en de eerste paar rijen van de dataset
# samples = rijen, kolommen=  features 

print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')
print(data.head())

# Checken of er missende waarden zijn 
# Er zijn geen missende waarden in deze dataset! 
print("\nMissing values per column:")
print(data.isnull().sum())

print("\nTotal missing values:")
print(data.isnull().sum().sum())

#%% Kijken hoe de dataset is opgebouwd  

# Kolomnamen opvragen 
# laatste kolomnaam is 'label'
print("\nColumn names:")
print(data.columns)

# Hoeveel ECG zijn abnormaal en hoeveel zijn normaal?

# Aangezien de laatste kolom de label is kunnen we het als volgt visualiseren 
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print("\nUnieke klassen:")
print(y.unique())
# Er zijn 2 klassen, namelijk 0 en 1.
# 0 is normaal, 1 is abnormaal.
# Er zijn 681 =0 en 146 = 1, dus bij 0 voorspellen sws hoge accuracy 
print(y.value_counts())

#%% train/test split, omdat je niet kan testen op training data.
# 80% trainen, 20% testen, willen we dit anders?? 
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Zorgt ervoor dat verdeling 0 en 1 gelijk blijft, want we hadden niet goed verdeelde data 
)

#%% Scaling, of onderstaand of wellicht SVM gebruiken?  

# nog kijken of we geen robust scaler willen toepassen 
scaler = RobustScaler()
# Fit alleen op training data
X_train_scaled = scaler.fit_transform(X_train)
# Gebruik dezelfde schaal op test data
X_test_scaled = scaler.transform(X_test) #geen fit transform op test want dan zou je test data lekken naar training data

#%% Feautre selectie 
# Dit voorkomt overfitting, en maakt het model eenvoudiger en sneller.
# Er zitten namelijk 9000 features in en 837 patienten (zie geprinte info)
#%% PCA - Principle component analysis 

#from sklearn.decomposition import PCA

# aantal componenten kiezen dat behouden moet worden hier moeten uiteindelijk de gekozen parameters in gevuld worden
pca = PCA(n_components = 0.95) 
#toepassen ( fitten en transformen op train data)
X_pca_train = pca.fit_transform(X_train_scaled)

# alleen transformen op test data 
X_pca_test = pca.transform(X_test_scaled) 

print(f"Aantal features voor PCA: {X_train_scaled.shape[1]}")
print(f"Aantal componenten na PCA (95% variantie): {X_pca_train.shape[1]}")


#%%pipeline maken voor PCA

from sklearn.model_selection import GridSearchCV

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



#%% pipeline PCA feature selectie op clf KNN
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
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
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


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
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

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





# %% Lasso mbv logistic regression
# juiste penalty en juiste Ltype kiezen 
# solver is de 'rekenmachine' die de juiste gewichten voor je features berekend. liblinear is specifiek voor kleine datasets. 
# importeren wanneer nodig 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel


lasso_select = SelectFromModel(LogisticRegression(penalty='l1', solver = 'liblinear', C=1.0 , class_weight= 'balanced', random_state= 42))

lasso_select.fit(X_train_scaled, y_train)

X_lasso_train = lasso_select.transform(X_train_scaled)
X_lasso_test = lasso_select.transform(X_test_scaled)

n_features_lasso = X_lasso_train.shape[1]
print(f"Aantal features voor Lasso: {X_train_scaled.shape[1]}")
print(f"Lasso heeft {n_features_lasso} over")



#%% Lasso mbv lasso 
from sklearn.linear_model import Lasso 
from sklearn.feature_selection import SelectFromModel

# 1. Definieer het basis model
lasso = Lasso(alpha=0.1)

# 2. Gebruik SelectFromModel om de selectie-logica toe te voegen
selector = SelectFromModel(estimator=lasso)

# 3. Fit de selector op de data
selector.fit(X_train_scaled, y_train)

# 4. Nu kun je WEL transformeren
X_lasso_train = selector.transform(X_train_scaled)
X_lasso_test = selector.transform(X_test_scaled)

n_features_lasso = X_lasso_train.shape[1]
print(f"Aantal features voor Lasso: {X_train_scaled.shape[1]}")
print(f"Lasso heeft {n_features_lasso} overgehouden")




# %% pipeline maken voor lasso 
# te importen packages
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso  
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
# vaste instellingen 
fixed_scaler = RobustScaler() 

# 1. Pipeline definitie
# We geven Lasso() mee aan SelectFromModel
lasso_par_pipe = Pipeline([
    ('scaler', RobustScaler()), 
    ('selector', SelectFromModel(Lasso())),
    ('clf', SVC(kernel='rbf', probability=True, class_weight='balanced'))
])

# 2. Parameter grid definitie
# BELANGRIJK: Gebruik selector__estimator__[parameter_naam]
param_grid = {
    'selector__estimator__alpha': [0.01, 0.1, 1.0, 10.0],
    'selector__estimator__max_iter': [1000, 3000, 5000],
    'selector__estimator__warm_start': [True, False],
    'selector__estimator__positive': [True],
    'selector__estimator__selection': ['random', 'cyclic']
}


# aram_grid = {'pca__n_components' : [ 0.80, 0.90, 0.95], 
#               'pca__whiten' : [True, False], 
#               'pca__svd_solver' : ['auto']} #


# beste combinatie zoeken 
grid = GridSearchCV(lasso_par_pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=1 )
grid.fit(X_train, y_train) 

print(f"Beste parameters: {grid.best_params_}")
print(f"Beste Cross-Validatie AUC: {grid.best_score_:4f}")



# %% pipeline Lasso feature selection op KNN
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# vaste instellingen 
fixed_scaler = RobustScaler() 
fixed_classifier = KNeighborsClassifier(n_neighbors=3)

#pipeline maken
lasso_par_pipe = Pipeline([
    ('scaler', RobustScaler()), 
    # We gebruiken LogisticRegression met L1 als de 'motor' voor selectie
    ('selector', SelectFromModel(Lasso())),
    ('clf', fixed_classifier)
])

#te testen parameters definieren 
# n components -> nog geen onderbouwing 
# whiten -> 
# svd solver andere kunnen niet met n_components waardes
param_grid = {
    'selector__estimator__alpha': [0.01, 0.1, 1.0, 10.0],
    'selector__estimator__max_iter': [1000, 3000, 5000],
    'selector__estimator__warm_start': [True, False],
    'selector__estimator__positive': [True],
    'selector__estimator__selection': ['random', 'cyclic']
}


# beste combinatie zoeken 
grid = GridSearchCV(lasso_par_pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=1 )
grid.fit(X_train, y_train) 

print(f"Beste parameters: {grid.best_params_}")
print(f"Beste Cross-Validatie AUC: {grid.best_score_:4f}")


#%% pipeline Lasso feature selection op random forest
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


# vaste instellingen 
fixed_scaler = RobustScaler() 
fixed_classifier = RandomForestClassifier (n_estimators = 100, random_state = 42)

#pipeline maken
lasso_par_pipe = Pipeline([
    ('scaler', RobustScaler()), # We gebruiken LogisticRegression met L1 als de 'motor' voor selectie
    ('selector', SelectFromModel(Lasso())),
    ('clf', fixed_classifier)])

#te testen parameters definieren 
# n components -> nog geen onderbouwing 
# whiten -> 
# svd solver andere kunnen niet met n_components waardes
param_grid = {
    'selector__estimator__alpha': [0.01, 0.1, 1.0, 10.0],
    'selector__estimator__max_iter': [1000, 3000, 5000],
    'selector__estimator__warm_start': [True, False],
    'selector__estimator__positive': [True],
    'selector__estimator__selection': ['random', 'cyclic']
}


# beste combinatie zoeken 
grid = GridSearchCV(lasso_par_pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=1 )
grid.fit(X_train, y_train) 

print(f"Beste parameters: {grid.best_params_}")
print(f"Beste Cross-Validatie AUC: {grid.best_score_:4f}")

#%% pipeline Lasso feature selection op XGBoost
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb


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
    ('selector', SelectFromModel(Lasso())),
    ('clf', fixed_classifier)])

#te testen parameters definieren 
# n components -> nog geen onderbouwing 
# whiten -> 
# svd solver andere kunnen niet met n_components waardes
param_grid = {
    'selector__estimator__alpha': [0.01, 0.1, 1.0, 10.0],
    'selector__estimator__max_iter': [1000, 3000, 5000],
    'selector__estimator__warm_start': [True, False],
    'selector__estimator__positive': [True],
    'selector__estimator__selection': ['random', 'cyclic']
}

# beste combinatie zoeken 
grid = GridSearchCV(lasso_par_pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=1 )
grid.fit(X_train, y_train) 

print(f"Beste parameters: {grid.best_params_}")
print(f"Beste Cross-Validatie AUC: {grid.best_score_:4f}")
# %%