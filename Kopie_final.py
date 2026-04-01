#%% ==================================================================================
# Basis
# ====================================================================================

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
from sklearn.metrics import (classification_report, roc_auc_score, average_precision_score,confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc)
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import cross_validate

#%% Loading data

def load_data():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(this_directory,'ecg' ,'ecg_data', 'ecg_data.csv'), index_col=0) # Standaard is None. Nu op 0. Dit betekent dat de eerste kolom als index wordt gebruikt.
    return data

data = load_data()

#%% Kijken hoe de dataset is opgebouwd 

print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')
print(data.head()) # Eerste paar regels laten zien

print("\nMissing values per column:")
print(data.isnull().sum()) 

print("\nTotal missing values:")
print(data.isnull().sum().sum())

print("\nColumn names:")
print(data.columns)

X = data.iloc[:, :-1] # alle rijen en alle kollommen behalve de laatste
y = data.iloc[:, -1] # alle rijen en alleen de laatste kolom

print("\nUnieke klassen:")
print(y.unique())

print(y.value_counts()) # geeft aan hoe de data is verdeeld. hoe vaak elke unieke waarde voorkomt. 

#%% Data opsplitsen in train en test

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42, # Een random state invullen. Maakt niet uit wat dit is. 42 is een grapje. "Antwoord op alles is 42".
    stratify=y  # Zorgt ervoor dat verdeling 0 en 1 gelijk blijft, want we hadden niet goed verdeelde data 
)

#%% Algemene dingetjes

# Voor class imbalance
n_negative = (y_train == 0).sum() # y train optellen met 0 
n_positive = (y_train == 1).sum() # y train optellen met 1
imbalance_ratio = n_negative / n_positive # Dit hebben we later nodig voor XGBoost. 

# Elke keer dat je de ene klasse, die minder vaak voorkomt, fout rekent, rekent dat bijvoorbeeld 10x zwaarder 
# mee dan wanneer je de klasse fout classificeerd die vaker voorkomt. Zo compenseer je voor class imbalance.


#%% ==================================================================================
# Logistic Regression + PCA 
# ====================================================================================

logreg_pca_pipe = Pipeline([
    ('scaler', StandardScaler()), 
    ('pca', PCA(n_components=0.8, svd_solver='auto', whiten=True)), 
    # PCA = Data in grafiek. Ze kijken waar het middelpunt ligt. Dat maken ze nieuwe 0.0 punt. Dan maken ze lijn in welke richting de data loopt. Dat is PC1. Dan zetten ze een lijn daarop 
    # loodrecht. Dat is PC2. En dan pakken ze ook nog PC3 en dat is ruis. Je maakt nieuwe features van hoog dimensionaal naar laag dimensionaal zonder informatie kwijt te raken.
    # n_components=0.8 --> hou genoeg componenten zodat 80% van de variatie behouden blijft. 
    # svd_solver='auto' --> SVD = singular value decomposition. SVD is de rekenmethode die PCA gebruikt om de belangrijkste richtingen (principal components) in je data te vinden.
    # whiten=True --> Whitening maakt alle PCA-componenten even groot qua spreiding, zodat geen enkele component domineert — maar je verliest wel info over welke richting het belangrijkst was. 
    # Dit maakt data verwerking ook sneller. 
    ('clf', LogisticRegression(class_weight='balanced', max_iter=3000, random_state=42)) 
    # Logistic regression =
    # class_weight='balanced' --> Fout in minderheidsklasse wordt zwaarder gewogen.
    # max_iter=3000 --> Hoe vaak je de weights mag optimaliseren.
    # random_state= 42 --> Waar je initailiseert. Een random state invullen. Maakt niet uit wat dit is. 42 is een grapje. "Antwoord op alles is 42". 
])

logreg_pca_param_grid = {
    'clf__C': [0.01, 0.1, 1, 10, 100],
    # Regularisatie sterkte. Hoe lager de C, hoe meer lineair het model, want minder last van grote gewichten die het model heel erg gaan sturen. Specifieke features mogen niet meer zeggen 
    # over de data. Dat wordt sterker afgestraft. Dit is dus een simpeler model. Minder kans op overfitting. 
    'clf__penalty': ['l2']
    # l2 is naar 0 en l1 is op 0. Niet perse uithalen, want dat hebben we al gedaan.
}

inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Inner loop bepaald welke hyperparameters. Outer loop is performance meten.
# n_splits=5 --> 20% validatie en 80% training.
# shuffle=True --> Data shuffelen voordat het gesplit wordt. Bijvoorbeeld eerst zieke dan gezonde mensen. Dat zit dan hierdoor door elkaar.x

# inner loop 
logreg_pca_grid = GridSearchCV(
    estimator=logreg_pca_pipe,
    param_grid=logreg_pca_param_grid,
    scoring='average_precision', # uses precision recall AUC. Onze dataset is imbalanced. Daarom belangrijk. PR AUC focust op de minority class. ROC AUC kijkt naar het algemene ding. 
    cv=inner_cv,
    n_jobs=-1, # bepaald waar je cpu zich op richt, 1 = op 1 cpu werken, minder werk en lagere werkkracht, -1 = alle mannen aan boord we gaan dit samen cheffen.  
    verbose=1 # prints progress updates while running so you can see it is working :)
)


# outer loop
logreg_pca_nested_scores = cross_validate(
    estimator=logreg_pca_grid, # the grid search itself is what gets evaluated
    X=X_train,
    y=y_train,
    cv=outer_cv,
    scoring='average_precision',
    n_jobs=1, # omdat andere al -1 gebruikt, dan gaat je computer naar de tyfus
    return_estimator=True, # opgeslagen welke waardes die heeft ingezet en dan makkelijk later inspecteren.
    return_train_score=False # trainingsscores niet returnen. 
)

print("Outer fold PR-AUC scores (LogReg + PCA):")
print(logreg_pca_nested_scores['test_score']) 
print(f"\nMean nested PR-AUC: {np.mean(logreg_pca_nested_scores['test_score']):.4f}")
print(f"Std nested PR-AUC: {np.std(logreg_pca_nested_scores['test_score']):.4f}")

#%% ==================================================================================
# SVM RBF + PCA
# ====================================================================================

svm_rbf_pca_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.8, svd_solver='auto', whiten=True)),
    ('clf', SVC(kernel='rbf', probability=True))
    # SVC = Support Vector Classifier.
    # Probeert een scheidingslijn (of vlak) te vinden tussen klassen, waarbij de afstand tot de dichtstbijzijnde punten (support vectors) maximaal is.
    # kernel='rbf' -->  RBF = radial basis function. Dit zorgt ervoor dat je geen rechte lijn meer trekt, maar een gebogen grens. Handig als je data niet lineair te scheiden is (wat vaak zo 
    # is in echte data). Je projecteert de data eigenlijk naar een hogere dimensie zodat het daar wel lineair scheidbaar wordt.
    # probability=True -->  Zorgt ervoor dat je niet alleen een klasse krijgt, maar ook een kans (bijv. 0.8 kans op klasse 1). Dit kost wel extra rekentijd, want er wordt een extra stap 
    # gedaan (Platt scaling). Belangrijk voor AUC.
])

svm_rbf_pca_param_grid = {
    'clf__C': [1e-2, 1e-1, 1],
    # lage C = minder overfitting, simpeler model
    # hoge C = meer kans op overfitting, maar betere fit op training data
    'clf__gamma': [1e-4, 1e-3, 1e-2]
    # Bepaalt hoe "ver" een datapunt invloed heeft. Hoever ruimte die om zich heen beïnvloedt. 
    # Lage gamma = elk punt kijkt ver om zich heen → smooth, globale beslisgrens
    # Hoge gamma = elk punt kijkt alleen heel lokaal → grillige, complexe grens

    # Kleinere C en kleinere gamma, want we hebben al eerdere genormaliseerd en gefit bij PCA. Dus extra zal alleen maar overfitting opleveren.
}

svm_rbf_pca_grid = GridSearchCV(
    estimator=svm_rbf_pca_pipe,
    param_grid=svm_rbf_pca_param_grid,
    scoring='average_precision',
    cv=inner_cv,
    n_jobs=-1,
    verbose=1
)

svm_rbf_pca_nested_scores = cross_validate(
    estimator=svm_rbf_pca_grid,
    X=X_train,
    y=y_train,
    cv=outer_cv,
    scoring='average_precision',
    n_jobs=1,
    return_train_score=False,
    return_estimator=True
)

print("Outer fold PR-AUC scores:")
print(svm_rbf_pca_nested_scores['test_score'])
print(f"\nGemiddelde nested PR-AUC: {svm_rbf_pca_nested_scores['test_score'].mean():.4f}")
print(f"Standaarddeviatie nested PR-AUC: {svm_rbf_pca_nested_scores['test_score'].std():.4f}")
print("\nBeste parameters per outer fold:")
for i, est in enumerate(svm_rbf_pca_nested_scores['estimator'], 1):
    print(f"Fold {i}: {est.best_params_}")

#%% ==================================================================================
# K Nearest Neighbours (KNN) + PCA
# ====================================================================================

knn_pca_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.8, svd_solver='auto', whiten=True)),
    ('clf', KNeighborsClassifier())
    # KNeighborsClassifier = Kijkt naar de k dichtstbijzijnde punten (neighbors) in de dataset en bepaalt op basis daarvan de klasse.
    # Dus: "bij welke groep lijken mijn buren te horen?"
])

knn_pca_param_grid = {
    'clf__n_neighbors': [3, 5, 9, 15, 20],
    # clf__n_neighbors --> Hoeveel buren je meeneemt in de beslissing.
    # Lage k (bijv. 3) --> heel gevoelig voor ruis → kan overfitten
    # Hoge k (bijv. 20) --> kijkt meer globaal → stabieler maar kan belangrijke details missen
    'clf__weights': ['uniform', 'distance']
    # clf__weights --> Hoe de buren meetellen:

    # 'uniform' --> 
    # Alle buren tellen even zwaar mee
    # Dus: gewoon meerderheid wint

    # 'distance' --> 
    # Dichtere buren tellen zwaarder mee dan verder weg
    # Dus: punten dichtbij hebben meer invloed (vaak logischer)
}

knn_pca_grid = GridSearchCV(
    estimator=knn_pca_pipe,
    param_grid=knn_pca_param_grid,
    scoring='average_precision',
    cv=inner_cv,
    n_jobs=-1,
    verbose=1
)

knn_pca_nested_scores = cross_validate(
    estimator=knn_pca_grid,
    X=X_train,
    y=y_train,
    cv=outer_cv,
    scoring='average_precision',
    n_jobs=1,
    return_estimator=True,
    return_train_score=False
)

print("Outer fold PR-AUC scores:")
print(knn_pca_nested_scores['test_score'])
print(f"\nMean nested PR-AUC: {np.mean(knn_pca_nested_scores['test_score']):.4f}")
print(f"Std nested PR-AUC: {np.std(knn_pca_nested_scores['test_score']):.4f}")
print("\nBest params per outer fold:")
for i, est in enumerate(knn_pca_nested_scores['estimator'], 1):
    print(f"Fold {i}: {est.best_params_}")


#%% ==================================================================================
# Random Forest + PCA
# ====================================================================================

rf_pca_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.8, svd_solver='auto', whiten=True)),
    ('clf', RandomForestClassifier(random_state=42, class_weight='balanced_subsample'))
    # RandomForestClassifier = Bestaat uit heel veel decision trees. Elke tree maakt een voorspelling en samen stemmen ze → majority vote.
    # Hierdoor minder gevoelig voor overfitting dan één enkele decision tree.
    # class_weight='balanced_subsample' --> Zorgt dat elke tree rekening houdt met class imbalance. Minderheidsklasse krijgt meer gewicht (maar per bootstrap sample bepaald).
])

rf_pca_param_grid = {
    'clf__n_estimators': [100, 200],
    # clf__n_estimators --> Aantal bomen in het bos. Meer bomen → stabielere voorspelling (maar kost meer tijd). Vaak: meer = beter, tot een bepaald punt.
    'clf__max_depth': [3, 5, 10], 
    # clf__max_depth --> Hoe diep elke boom mag groeien.
    # Lage depth → simpele bomen → minder overfitting
    # Hoge depth → complexe bomen → meer kans op overfitting
    'clf__max_features': ['sqrt'],
    # clf__max_features --> Hoeveel features elke split mag bekijken. 'sqrt' = √(aantal features). Zorgt voor variatie tussen bomen → minder overfitting. 
    'clf__min_samples_split': [2, 5],
    # clf__min_samples_split --> Minimum aantal samples nodig om een split te maken.
    # Hoger → minder snel splitsen → simpelere bomen
    'clf__min_samples_leaf': [2, 4]
    # clf__min_samples_leaf --> Minimum aantal samples in een leaf (eindpunt van boom).
    # Hoger → voorkomt dat bladeren gebaseerd zijn op weinig data → minder overfitting
}

rf_pca_grid = GridSearchCV(
    estimator=rf_pca_pipe,
    param_grid=rf_pca_param_grid,
    scoring='average_precision',
    cv=inner_cv,
    n_jobs=-1,
    verbose=1
)

rf_pca_nested_scores = cross_validate(
    estimator=rf_pca_grid,
    X=X_train,
    y=y_train,
    cv=outer_cv,
    scoring='average_precision',
    n_jobs=1,
    return_estimator=True,
    return_train_score=False
)

print("Outer fold PR-AUC scores:")
print(rf_pca_nested_scores['test_score'])
print(f"\nMean nested PR-AUC: {np.mean(rf_pca_nested_scores['test_score']):.4f}")
print(f"Std nested PR-AUC: {np.std(rf_pca_nested_scores['test_score']):.4f}")
print("\nBest params per outer fold:")
for i, est in enumerate(rf_pca_nested_scores['estimator'], 1):
    print(f"Fold {i}: {est.best_params_}")


#%% ==================================================================================
# XGBoost + PCA
# ====================================================================================

xgb_pca_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.8, svd_solver='auto', whiten=True)),
    ('clf', xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        scale_pos_weight=imbalance_ratio,
        random_state=42
    ))
    # XGBClassifier = Boosting model (anders dan Random Forest). Bomen worden NIET onafhankelijk gemaakt, maar na elkaar. Elke nieuwe boom probeert de fouten van de vorige bomen te corrigeren.
    
    # objective='binary:logistic' --> Voor binaire classificatie (output = kans tussen 0 en 1)

    # eval_metric='auc' --> Model wordt geëvalueerd op ROC-AUC

    # scale_pos_weight=imbalance_ratio --> Minderheidsklasse krijgt meer gewicht (belangrijk bij scheve data)
])

xgb_pca_param_grid = {
    'clf__n_estimators': [100],
    # clf__n_estimators --> Aantal bomen (boosting stappen)
    # Meer bomen → beter leren, maar ook kans op overfitting
    'clf__max_depth': [2, 3],          # lager!
    # clf__max_depth --> Hoe diep elke boom mag zijn
    # Hier bewust laag (2–3) → simpele bomen → minder overfitting
    # Boosting werkt vaak beter met veel simpele bomen
    'clf__learning_rate': [0.05],
    # clf__learning_rate --> Hoeveel elke boom bijdraagt
    # Lage learning rate → langzamer leren → vaak beter resultaat (maar dan heb je meestal meer bomen nodig)
    'clf__subsample': [0.6, 0.8],      # meer regularisatie
    # clf__subsample --> Percentage van data dat elke boom gebruikt
    # < 1 → random subset → minder overfitting
    'clf__colsample_bytree': [0.6, 0.8]
    # clf__colsample_bytree --> Percentage features per boom
    # Zorgt voor extra randomisatie → betere generalisatie
}


xgb_pca_grid = GridSearchCV(
    estimator=xgb_pca_pipe,
    param_grid=xgb_pca_param_grid,
    scoring='average_precision',
    cv=inner_cv,
    n_jobs=-1,
    verbose=1
)

xgb_pca_nested_scores = cross_validate(
    estimator=xgb_pca_grid,
    X=X_train,
    y=y_train,
    cv=outer_cv,
    scoring='average_precision',
    n_jobs=1,
    return_estimator=True,
    return_train_score=False
)

print("Outer fold PR-AUC scores:")
print(xgb_pca_nested_scores['test_score'])
print(f"\nMean nested PR-AUC: {np.mean(xgb_pca_nested_scores['test_score']):.4f}")
print(f"Std nested PR-AUC: {np.std(xgb_pca_nested_scores['test_score']):.4f}")
print("\nBest params per outer fold:")
for i, est in enumerate(xgb_pca_nested_scores['estimator'], 1):
    print(f"Fold {i}: {est.best_params_}")

#%% ==================================================================================
# Linear SVM + PCA 
# ====================================================================================

linear_svm_pca_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.8, svd_solver='auto', whiten=True)),
    ('clf', SVC(kernel='linear', class_weight='balanced', probability=True, random_state=42))
])

linear_svm_pca_param_grid = {
    'clf__C': [0.01, 0.1, 1, 10]
}

linear_svm_pca_grid = GridSearchCV(
    estimator=linear_svm_pca_pipe,
    param_grid=linear_svm_pca_param_grid,
    scoring='average_precision',
    cv=inner_cv,
    n_jobs=-1,
    verbose=1
)

linear_svm_pca_nested_scores = cross_validate(
    estimator=linear_svm_pca_grid,
    X=X_train,
    y=y_train,
    cv=outer_cv,
    scoring='average_precision',
    n_jobs=1,
    return_estimator=True,
    return_train_score=False
)

print("Outer fold PR-AUC scores (Linear SVM + PCA):")
print(linear_svm_pca_nested_scores['test_score'])
print(f"\nMean nested PR-AUC: {np.mean(linear_svm_pca_nested_scores['test_score']):.4f}")
print(f"Std nested PR-AUC: {np.std(linear_svm_pca_nested_scores['test_score']):.4f}")

#%% ==================================================================================
# SVM RBF + logreg
# ====================================================================================

svm_rbf_logreg_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectFromModel(
        LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', C=0.1, random_state=42, max_iter=3000)
    )),
    # # SelectFromModel = Gebruikt een model om te bepalen welke features belangrijk zijn, en gooit de minder belangrijke features weg. 
    # Hier gebruik je Logistic Regression als "feature selector".

    # penalty='l1' --> L1 regularisatie zet sommige gewichten EXACT op 0.
    # → features met gewicht 0 worden verwijderd
    # → automatische feature selectie

    # solver='liblinear' --> Nodig om L1 te kunnen gebruiken (niet elke solver kan dat)

    # class_weight='balanced' --> Houdt rekening met class imbalance bij het bepalen van belangrijke features

    # C=0.1 --> Lage C = sterke regularisatie
    # → meer gewichten gaan naar 0
    # → dus: strengere feature selectie

    # max_iter=3000 --> Genoeg iteraties om te convergeren
    ('clf', SVC(kernel='rbf', probability=True))
])

svm_rbf_logreg_param_grid = {
    'clf__C': [1e-2, 1e-1, 1, 10, 100],
    'clf__gamma': [1e-4, 1e-3, 1e-2, 1e-1, 1]
}

svm_rbf_logreg_grid = GridSearchCV(
    estimator=svm_rbf_logreg_pipe,
    param_grid=svm_rbf_logreg_param_grid,
    scoring='average_precision',
    cv=inner_cv,
    n_jobs=-1,
    verbose=1
)

svm_rbf_logreg_nested_scores = cross_validate(
    estimator=svm_rbf_logreg_grid,
    X=X_train,
    y=y_train,
    cv=outer_cv,
    scoring='average_precision',
    n_jobs=1,
    return_train_score=False,
    return_estimator=True
)

print("Outer fold PR-AUC scores:")
print(svm_rbf_logreg_nested_scores['test_score'])
print(f"\nGemiddelde nested PR-AUC: {svm_rbf_logreg_nested_scores['test_score'].mean():.4f}")
print(f"Standaarddeviatie nested PR-AUC: {svm_rbf_logreg_nested_scores['test_score'].std():.4f}")
print("\nBeste parameters per outer fold:")
for i, est in enumerate(svm_rbf_logreg_nested_scores['estimator'], 1):
    print(f"Fold {i}: {est.best_params_}")


#%% ==================================================================================
# KNN + logreg
# ====================================================================================

knn_logreg_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectFromModel(
        LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', C=0.1, random_state=42, max_iter=3000)
    )),
    ('clf', KNeighborsClassifier())
])

knn_logreg_param_grid = {
    'clf__n_neighbors': [3, 5, 9, 15, 20],
    'clf__weights': ['uniform', 'distance']
}

knn_logreg_grid = GridSearchCV(
    estimator=knn_logreg_pipe,
    param_grid=knn_logreg_param_grid,
    scoring='average_precision',
    cv=inner_cv,
    n_jobs=-1,
    verbose=1
)

knn_logreg_nested_scores = cross_validate(
    estimator=knn_logreg_grid,
    X=X_train,
    y=y_train,
    cv=outer_cv,
    scoring='average_precision',
    n_jobs=1,
    return_estimator=True,
    return_train_score=False
)

print("Outer fold PR-AUC scores:")
print(knn_logreg_nested_scores['test_score'])
print(f"\nMean nested PR-AUC: {np.mean(knn_logreg_nested_scores['test_score']):.4f}")
print(f"Std nested PR-AUC: {np.std(knn_logreg_nested_scores['test_score']):.4f}")
print("\nBest params per outer fold:")
for i, est in enumerate(knn_logreg_nested_scores['estimator'], 1):
    print(f"Fold {i}: {est.best_params_}")


#%% ==================================================================================
# Random Forest + logreg
# ====================================================================================

rf_logreg_pipe = Pipeline([
    ('scaler',  StandardScaler()),
    ('selector', SelectFromModel(
        LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', C=0.1, random_state=42, max_iter=3000)
    )),
    ('clf', RandomForestClassifier(random_state=42, class_weight='balanced_subsample'))
])

rf_logreg_param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [3, 5, 10], 
    'clf__max_features': ['sqrt'],
    'clf__min_samples_split': [2, 5],
    'clf__min_samples_leaf': [2, 4] 
}

rf_logreg_grid = GridSearchCV(
    estimator=rf_logreg_pipe,
    param_grid=rf_logreg_param_grid,
    scoring='average_precision',
    cv=inner_cv,
    n_jobs=-1,
    verbose=1
)

rf_logreg_nested_scores = cross_validate(
    estimator=rf_logreg_grid,
    X=X_train,
    y=y_train,
    cv=outer_cv,
    scoring='average_precision',
    n_jobs=1,
    return_estimator=True,
    return_train_score=False
)

print("Outer fold PR-AUC scores:")
print(rf_logreg_nested_scores['test_score'])
print(f"\nMean nested PR-AUC: {np.mean(rf_logreg_nested_scores['test_score']):.4f}")
print(f"Std nested PR-AUC: {np.std(rf_logreg_nested_scores['test_score']):.4f}")
print("\nBest params per outer fold:")
for i, est in enumerate(rf_logreg_nested_scores['estimator'], 1):
    print(f"Fold {i}: {est.best_params_}")


#%% ==================================================================================
# XGBoost + logreg
# ====================================================================================

xgb_logreg_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectFromModel(
        LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', C=0.1, random_state=42, max_iter=3000)
    )),
    ('clf', xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        scale_pos_weight=imbalance_ratio,
        random_state=42
    ))
])

xgb_logreg_param_grid = {
    'clf__n_estimators': [100, 200], 
    'clf__max_depth': [2, 3, 5], 
    'clf__learning_rate': [0.05, 0.1],
    'clf__subsample': [0.8],
    'clf__colsample_bytree': [0.8] 
}

xgb_logreg_grid = GridSearchCV(
    estimator=xgb_logreg_pipe,
    param_grid=xgb_logreg_param_grid,
    scoring='average_precision',
    cv=inner_cv,
    n_jobs=-1,
    verbose=1
)

xgb_logreg_nested_scores = cross_validate(
    estimator=xgb_logreg_grid,
    X=X_train,
    y=y_train,
    cv=outer_cv,
    scoring='average_precision',
    n_jobs=1,
    return_estimator=True,
    return_train_score=False
)

print("Outer fold PR-AUC scores:")
print(xgb_logreg_nested_scores['test_score'])
print(f"\nMean nested PR-AUC: {np.mean(xgb_logreg_nested_scores['test_score']):.4f}")
print(f"Std nested PR-AUC: {np.std(xgb_logreg_nested_scores['test_score']):.4f}")
print("\nBest params per outer fold:")
for i, est in enumerate(xgb_logreg_nested_scores['estimator'], 1):
    print(f"Fold {i}: {est.best_params_}")

#%% ==================================================================================
# Linear SVM + logreg 
# ====================================================================================

svm_linear_logreg_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectFromModel(
        LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', C=0.1, random_state=42, max_iter=3000)
    )),
    ('clf', SVC(kernel='linear', class_weight='balanced', probability=True))
])

svm_linear_logreg_param_grid = {
    'selector__estimator__C': [0.05, 0.1, 0.5], 
    'clf__C': [0.01, 0.1, 1, 10]
}

svm_linear_logreg_grid = GridSearchCV(
    estimator=svm_linear_logreg_pipe,
    param_grid=svm_linear_logreg_param_grid,
    scoring='average_precision',
    cv=inner_cv,
    n_jobs=-1,
    verbose=1
)

svm_linear_logreg_nested_scores = cross_validate(
    estimator=svm_linear_logreg_grid,
    X=X_train,
    y=y_train,
    cv=outer_cv,
    scoring='average_precision',
    n_jobs=1,
    return_estimator=True,
    return_train_score=False
)

#%% ==================================================================================
# Vergelijking van geselecteerde pipelines + keuze beste pipeline + test op unseen set
# ====================================================================================

#%% ----------------------------------------------------------------------------------
# 1. Nested CV resultaten verzamelen van de pipelines die we willen vergelijken
# -----------------------------------------------------------------------------------

selected_pipelines = {
    'SVM RBF + PCA': {
        'nested_scores': svm_rbf_pca_nested_scores['test_score'],
        'grid': svm_rbf_pca_grid
    },
    'KNN + PCA': {
        'nested_scores': knn_pca_nested_scores['test_score'],
        'grid': knn_pca_grid
    },
    'Random Forest + PCA': {
        'nested_scores': rf_pca_nested_scores['test_score'],
        'grid': rf_pca_grid
    },
    'XGBoost + PCA': {
        'nested_scores': xgb_pca_nested_scores['test_score'],
        'grid': xgb_pca_grid
    },
    'SVM RBF + log-Reg': {
        'nested_scores': svm_rbf_logreg_nested_scores['test_score'],
        'grid': svm_rbf_logreg_grid
    },
    'KNN + log-Reg': {
        'nested_scores': knn_logreg_nested_scores['test_score'],
        'grid': knn_logreg_grid
    },
    'Random Forest + log-Reg': {
        'nested_scores': rf_logreg_nested_scores['test_score'],
        'grid': rf_logreg_grid
    },
    'XGBoost + log-Reg': {
        'nested_scores': xgb_logreg_nested_scores['test_score'],
        'grid': xgb_logreg_grid
    }
}

# selected_pipelines = Dictionary waarin we per pipeline opslaan:
# 1. de nested CV test scores van de outer folds
# 2. de bijbehorende GridSearchCV, zodat we later de beste pipeline opnieuw kunnen fitten

#%% ----------------------------------------------------------------------------------
# 2. Tabel maken met alle nested CV PR-AUC scores per outer fold
# -----------------------------------------------------------------------------------

nested_scores_table = pd.DataFrame({
    pipeline_name: np.array(pipeline_info['nested_scores'])
    for pipeline_name, pipeline_info in selected_pipelines.items()
})

nested_scores_table.index = [f'Fold {i}' for i in range(1, len(nested_scores_table) + 1)]

print("\n" + "="*100)
print("TABEL MET NESTED CV PR-AUC SCORES PER OUTER FOLD")
print("="*100)
print(nested_scores_table)

# Hier maken we een overzichtstabel:
# rijen = outer folds
# kolommen = pipelines
# waarden = PR-AUC score op die specifieke outer test fold

# Zo kun je pipeline-prestaties direct per fold vergelijken.

#%% ----------------------------------------------------------------------------------
# 3. Samenvattingstabel maken met mean/std/min/max per pipeline
# -----------------------------------------------------------------------------------

comparison_rows = []

for pipeline_name, pipeline_info in selected_pipelines.items():
    scores = np.array(pipeline_info['nested_scores'])

    comparison_rows.append({
        'Pipeline': pipeline_name,
        'Mean PR-AUC': scores.mean(),
        'Std PR-AUC': scores.std(),
        'Min PR-AUC': scores.min(),
        'Max PR-AUC': scores.max()
    })

comparison_df = pd.DataFrame(comparison_rows)

comparison_df = comparison_df.sort_values(
    by=['Mean PR-AUC', 'Std PR-AUC'],
    ascending=[False, True]
).reset_index(drop=True)

comparison_df['Mean PR-AUC'] = comparison_df['Mean PR-AUC'].round(3)
comparison_df['Std PR-AUC'] = comparison_df['Std PR-AUC'].round(3)
comparison_df['Min PR-AUC'] = comparison_df['Min PR-AUC'].round(3)
comparison_df['Max PR-AUC'] = comparison_df['Max PR-AUC'].round(3)

print("\n" + "="*100)
print("TABEL MET SAMENVATTING VAN DE NESTED CV RESULTATEN (PR-AUC)")
print("="*100)
print(comparison_df)

# Hier vatten we per pipeline de prestaties samen.
# Mean PR-AUC = gemiddelde prestatie over alle outer folds
# Std PR-AUC = hoeveel de prestaties variëren tussen folds
# Min/Max = slechtste en beste foldscore

# Daarna sorteren we:
# eerst op hoogste mean PR-AUC
# en bij gelijke mean op laagste std
# --> dus: liefst goed én stabiel

#%% Beste pipeline expliciet tonen
best_pipeline_name = comparison_df.loc[0, 'Pipeline']

print("\nBeste pipeline op basis van nested CV resultaten:")
print(best_pipeline_name)

#%% Compacte tabel voor verslag
comparison_df_compact = comparison_df[['Pipeline', 'Mean PR-AUC', 'Std PR-AUC']].copy()

print("\n" + "="*100)
print("COMPACTE TABEL VOOR VERGELIJKING VAN DE PIPELINES")
print("="*100)
print(comparison_df_compact)

#%% ----------------------------------------------------------------------------------
# 4. Plot 1: lijnplot van nested CV PR-AUC per outer fold
# -----------------------------------------------------------------------------------

plt.figure(figsize=(12, 7))

n_folds = len(next(iter(selected_pipelines.values()))['nested_scores'])
fold_numbers = np.arange(1, n_folds + 1)

for pipeline_name, pipeline_info in selected_pipelines.items():
    scores = np.array(pipeline_info['nested_scores'])
    plt.plot(
        fold_numbers,
        scores,
        marker='o',
        linewidth=2,
        label=pipeline_name
    )

plt.xlabel('Outer fold')
plt.ylabel('PR-AUC')
plt.title('Nested CV PR-AUC per outer fold for every pipeline')
plt.xticks(fold_numbers)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Deze plot laat per pipeline zien hoe de PR-AUC verandert over de outer folds.
# Hiermee zie je niet alleen het gemiddelde, maar ook of een model heel wisselend presteert tussen verschillende train/test splits.

#%% ----------------------------------------------------------------------------------
# Learning curve van de beste pipeline (alleen op trainingsdata)
# -----------------------------------------------------------------------------------

from sklearn.model_selection import learning_curve

learning_curve_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

train_sizes, train_scores, val_scores = learning_curve(
    estimator=best_final_model,
    X=X_train,
    y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 8),
    cv=learning_curve_cv,
    scoring='average_precision',
    n_jobs=-1,
    shuffle=True,
    random_state=42
)

# Gemiddelden en standaarddeviaties berekenen
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plotten
plt.figure(figsize=(8, 6))

plt.plot(train_sizes, train_mean, marker='o', linewidth=2, label='Train PR-AUC')
plt.fill_between(
    train_sizes,
    train_mean - train_std,
    train_mean + train_std,
    alpha=0.2
)

plt.plot(train_sizes, val_mean, marker='o', linewidth=2, label='Validation PR-AUC')
plt.fill_between(
    train_sizes,
    val_mean - val_std,
    val_mean + val_std,
    alpha=0.2
)

plt.xlabel('Aantal trainingssamples')
plt.ylabel('PR-AUC')
plt.title(f'Learning curve - {best_pipeline_name}')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


learning_curve_table = pd.DataFrame({
    'Train size': train_sizes,
    'Train mean PR-AUC': train_mean,
    'Train std PR-AUC': train_std,
    'Validation mean PR-AUC': val_mean,
    'Validation std PR-AUC': val_std
})

print("\n" + "="*100)
print("LEARNING CURVE RESULTATEN")
print("="*100)
print(learning_curve_table.round(4))
#%% ----------------------------------------------------------------------------------
# 5. Plot 2: gecombineerde boxplot + mean ± std in één figuur
# -----------------------------------------------------------------------------------

ordered_pipeline_names = comparison_df['Pipeline'].tolist()
ordered_scores = [np.array(selected_pipelines[name]['nested_scores']) for name in ordered_pipeline_names]
ordered_means = comparison_df['Mean PR-AUC'].values
ordered_stds = comparison_df['Std PR-AUC'].values

plt.figure(figsize=(13, 7))

box = plt.boxplot(
    ordered_scores,
    tick_labels=ordered_pipeline_names,
    patch_artist=True,
    widths=0.6
)

x_positions = np.arange(1, len(ordered_pipeline_names) + 1)

plt.errorbar(
    x_positions,
    ordered_means,
    yerr=ordered_stds,
    fmt='o',
    capsize=6,
    linewidth=2,
    label='Mean ± std'
)

plt.ylabel('PR-AUC')
plt.title('Boxplot of nested CV PR-AUC scores with mean ± std per pipeline')
plt.xticks(rotation=25, ha='right')
plt.grid(True, axis='y', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

#%% ----------------------------------------------------------------------------------
# 6. Beste pipeline kiezen op basis van de samenvattingstabel
# -----------------------------------------------------------------------------------

best_pipeline_name = comparison_df.loc[0, 'Pipeline']
best_pipeline_grid = selected_pipelines[best_pipeline_name]['grid']

print("\n" + "="*100)
print("BESTE PIPELINE GEKOZEN OP BASIS VAN NESTED CV")
print("="*100)
print(f"Beste pipeline: {best_pipeline_name}")
print(f"Mean nested PR-AUC: {comparison_df.loc[0, 'Mean PR-AUC']:.4f}")
print(f"Std nested PR-AUC:  {comparison_df.loc[0, 'Std PR-AUC']:.4f}")


#%% ----------------------------------------------------------------------------------
# 7. Beste pipeline opnieuw fitten op de volledige trainingsset, FINAL MODEL!!
# -----------------------------------------------------------------------------------

best_pipeline_grid.fit(X_train, y_train)

best_final_model = best_pipeline_grid.best_estimator_

print("\n" + "="*100)
print("DEFINITIEVE FIT VAN DE BESTE PIPELINE OP DE VOLLEDIGE TRAININGSSET")
print("="*100)
print(f"Gekozen pipeline: {best_pipeline_name}")
print("Beste hyperparameters op volledige trainingsset:")
print(best_pipeline_grid.best_params_)
print(f"Beste CV PR-AUC op volledige trainingsset: {best_pipeline_grid.best_score_:.4f}")


from sklearn.metrics import precision_recall_curve


y_train_proba = best_final_model.predict_proba(X_train)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_train, y_train_proba)

f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)

best_idx = np.argmax(f1_scores)
optimale_drempel = thresholds[best_idx]

print(f"\nDe optimaal berekende drempel op de trainingsdata is: {optimale_drempel:.3f}")
#%% ----------------------------------------------------------------------------------
# 8. Evaluatie op de unseen 20% testset
# -----------------------------------------------------------------------------------

y_test_proba = best_final_model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= optimale_drempel).astype(int)

test_roc_auc = roc_auc_score(y_test, y_test_proba)
test_pr_auc = average_precision_score(y_test, y_test_proba)

print("\n" + "="*100)
print("EVALUATIE OP DE UNSEEN TESTSET")
print("="*100)
print(f"Beste pipeline: {best_pipeline_name}")
print(f"Test ROC-AUC: {test_roc_auc:.4f}")
print(f"Test PR-AUC:  {test_pr_auc:.4f}")

print(f"\nClassification report op testset (met berekende drempel {optimale_drempel:.3f}):")
print(classification_report(y_test, y_test_pred))

#%% ----------------------------------------------------------------------------------
# 9. ROC-curve plotten op de testset
# -----------------------------------------------------------------------------------

fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
roc_curve_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f'{best_pipeline_name} (AUC = {roc_curve_auc:.3f})'
)
plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1, label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC-curve on the testset - {best_pipeline_name}')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%% ----------------------------------------------------------------------------------
# 9. Precision-Recall curve plotten op de testset
# -----------------------------------------------------------------------------------
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
pr_auc = average_precision_score(y_test, y_test_proba)

plt.figure(figsize=(8, 6))
plt.plot(
    recall,
    precision,
    linewidth=2,
    label=f'{best_pipeline_name} (PR-AUC = {pr_auc:.3f})'
)

no_skill = len(y_test[y_test == 1]) / len(y_test)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', linewidth=1, label=f'Random classifier (Baseline = {no_skill:.3f})')

plt.xlabel('Recall (True Positive Rate)')
plt.ylabel('Precision (Positive Predictive Value)')
plt.title(f'Precision-Recall curve on the testset - {best_pipeline_name}')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
#%% ----------------------------------------------------------------------------------
# 10. Confusion matrix plotten op de testset
# -----------------------------------------------------------------------------------

cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title(f'Confusion matrix on the testset (Drempel={aangepaste_drempel}) - {best_pipeline_name}')
plt.tight_layout()
plt.show()

#%% ----------------------------------------------------------------------------------
# 11. Extra samenvatting opslaan in variabelen / tabellen
# -----------------------------------------------------------------------------------

final_results_summary = {
    'best_pipeline_name': best_pipeline_name,
    'best_nested_cv_mean_pr_auc': comparison_df.loc[0, 'Mean PR-AUC'],
    'best_nested_cv_std_pr_auc': comparison_df.loc[0, 'Std PR-AUC'],
    'best_train_cv_params': best_pipeline_grid.best_params_,
    'best_train_cv_pr_auc': best_pipeline_grid.best_score_,
    'test_roc_auc': test_roc_auc,
    'test_pr_auc': test_pr_auc,
    'confusion_matrix': cm
}

print("\n" + "="*100)
print("Summary of Final Results")
print("="*100)
for key, value in final_results_summary.items():
    print(f"{key}: {value}")

#%%