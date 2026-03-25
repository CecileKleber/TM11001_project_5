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



def load_data():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(this_directory, 'ecg', 'ecg_data', 'ecg_data.csv'), index_col=0)
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
scaler = StandardScaler()
# Fit alleen op training data
X_train_scaled = scaler.fit_transform(X_train)
# Gebruik dezelfde schaal op test data
X_test_scaled = scaler.transform(X_test) #geen fit transform op test want dan zou je test data lekken naar training data

#%% Feautre selectie 
# Dit voorkomt overfitting, en maakt het model eenvoudiger en sneller.
# Er zitten namelijk 9000 features in en 837 patienten (zie gepinte info)

# Beste k vinden 
k_waardes = range(20, 1000, 4)

k_waardes = (list(k_waardes))

auc_scores = []
for k in k_waardes:
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    model = SVC(kernel='rbf', probability=True)
    model.fit(X_train_selected, y_train)
    
    y_prob = model.predict_proba(X_test_selected)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    
 
    auc_scores.append(auc)
    

# hier zoeken we naar de beste k
max_auc = max(auc_scores)                  # Zoek het hoogste getal in de lijst
best_index = auc_scores.index(max_auc)     # Op welke positie (index) staat dit getal?
best_k = k_waardes[best_index]
print(f"Beste k: {best_k} met ROC-AUC: {max_auc:.4f}")
# Beste k = bij 126 

# SelecKbest 
k = best_k # dit getal kunnen we aanpassen ahv wat bij de data past, wordt nu dus aangepast ahv de for loop
selector = SelectKBest(score_func=f_classif, k=k)
# Fit alleen op training
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
# Pas toe op test
X_test_selected = selector.transform(X_test_scaled)
print("Aantal features na selectie:", X_train_selected.shape[1])

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
    n_jobs=-1,
    verbose=1
)

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