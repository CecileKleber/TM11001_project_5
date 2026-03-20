#%%
# =============================================================================
# ECG Classification Pipeline
# Feature selection: PCA | LASSO (SelectFromModel)
# Classifiers:       SVM-RBF | KNN | Random Forest | XGBoost
# Evaluation:        Nested cross-validation (ROC-AUC)
#
# Run this file top-to-bottom (F5), or run each cell in order.
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, cross_validate
)
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
from matplotlib.patches import Patch

#%%
# DEBUG MODE
# Set DEBUG = True for a quick smoke-test (1 combination, tiny grids, 3 folds)
# Set DEBUG = False for the full run
# =============================================================================

DEBUG = False   # <--- flip to False for the real run

#%%
# 1. Load data
# =============================================================================

def load_data():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(this_directory, 'ecg_data.csv'), index_col=0)
    return data

data = load_data()

print(f"Samples:  {len(data.index)}")
print(f"Columns:  {len(data.columns)}")
print(f"\nMissing values: {data.isnull().sum().sum()}")
print(f"\nClass distribution:\n{data.iloc[:, -1].value_counts()}")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

#%%
# 2. Train/test split (held-out test set — only touched at the very end)
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

#%%
# 3. Cross-validation settings
# =============================================================================

n_folds = 3 if DEBUG else 5
inner_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
outer_cv  = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

print(f"DEBUG mode: {DEBUG}  |  CV folds: {n_folds}")

#%%
# 4. SelectKBest k-analysis
#    This uses a fixed SVM to find which k gives the best AUC.
#    It's a reference/comparison — the main pipeline uses PCA and LASSO instead.
#    Note: this uses simple CV (not nested), so the AUC will look a bit higher
#    than the nested CV results below. That's expected and not a bug.
# =============================================================================

k_values = [20, 50, 100, 150, 200, 300, 500, 1000]  # always full, this runs fast
k_scores = []

print("Running SelectKBest k-analysis …")
for k in k_values:
    pipe_k = Pipeline([
        ('scaler',   RobustScaler()),
        ('selector', SelectKBest(score_func=f_classif, k=k)),
        ('clf',      SVC(kernel='rbf', gamma='scale', C=1,
                         probability=True, class_weight='balanced'))
    ])
    scores = cross_val_score(pipe_k, X_train, y_train,
                             cv=inner_cv, scoring='roc_auc', n_jobs=-1)
    k_scores.append(scores.mean())
    print(f"  k = {k:4d} | mean CV ROC-AUC = {scores.mean():.4f}")

best_k = k_values[int(np.argmax(k_scores))]
print(f"\nBest k: {best_k}  (AUC = {max(k_scores):.4f})")

# Plot
plt.figure(figsize=(8, 4))
plt.plot(k_values, k_scores, marker='o', color='#2196F3')
plt.xlabel("Number of selected features (k)")
plt.ylabel("Mean CV ROC-AUC")
plt.title("SelectKBest k-analysis (reference — simple CV, not nested)")
plt.grid(True)
plt.tight_layout()
plt.show()


#%%
# 5. Define classifiers & their hyperparameter grids
# =============================================================================

# Class imbalance ratio for XGBoost
imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()

classifiers = {

    'SVM-RBF': {
        'clf': SVC(kernel='rbf', probability=True, class_weight='balanced'),
        'params': {
            'clf__C':     [0.1, 1] if DEBUG else [1e-2, 1e-1, 1, 10, 100],
            'clf__gamma': [1e-3, 1e-2] if DEBUG else [1e-4, 1e-3, 1e-2, 1e-1, 1]
        }
    },

    'SVM-Poly': {
        'clf': SVC(kernel='poly', probability=True, class_weight='balanced'),
        'params': {
            'clf__C':      [0.1, 1]    if DEBUG else [1e-2, 1e-1, 1, 10, 100],
            'clf__gamma':  [1e-3, 1e-2] if DEBUG else [1e-4, 1e-3, 1e-2, 1e-1, 1],
            'clf__degree': [2, 3]      if DEBUG else [2, 3, 4],
            'clf__coef0':  [0, 1]      if DEBUG else [0, 0.5, 1]
        }
    },

    'KNN': {
        'clf': KNeighborsClassifier(),
        'params': {
            'clf__n_neighbors': [3, 5] if DEBUG else [3, 5, 7, 11, 15, 19],
            'clf__weights':     ['uniform', 'distance'],
            'clf__p':           [2]    if DEBUG else [1, 2]
        }
    },

    'RandomForest': {
        'clf': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'params': {
            'clf__n_estimators':     [50]          if DEBUG else [100, 200, 500],
            'clf__max_depth':        [5]            if DEBUG else [None, 10, 20],
            'clf__max_features':     ['sqrt']       if DEBUG else ['sqrt', 'log2'],
            'clf__min_samples_leaf': [1]            if DEBUG else [1, 2, 4]
        }
    },

    'XGBoost': {
        'clf': xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            scale_pos_weight=imbalance_ratio,
            random_state=42
        ),
        'params': {
            'clf__n_estimators':     [50]        if DEBUG else [100, 200, 300],
            'clf__max_depth':        [3]         if DEBUG else [3, 5, 7],
            'clf__learning_rate':    [0.1]       if DEBUG else [0.01, 0.05, 0.1],
            'clf__subsample':        [0.8]       if DEBUG else [0.8, 1.0],
            'clf__colsample_bytree': [0.8]       if DEBUG else [0.8, 1.0]
        }
    }
}

#%%
# 6. Feature selectors — two options
#    Note: Random Forest is tree-based (no distance) so scaling is skipped
#          for the RF steps; all other classifiers include RobustScaler.
# =============================================================================

feature_selectors = {

    'PCA': {
        'steps': [
            ('scaler',   RobustScaler()),
            ('selector', PCA(n_components=0.95, svd_solver='full'))
        ],
        'extra_params': {}
    },

    'LASSO': {
        'steps': [
            ('scaler',   RobustScaler()),
            ('selector', SelectFromModel(
                LogisticRegression(
                    penalty='l1',
                    solver='liblinear',
                    class_weight='balanced',
                    random_state=42,
                    max_iter=1000
                )
            ))
        ],
        'extra_params': {
            'selector__estimator__C': [0.1, 1.0] if DEBUG else [0.01, 0.1, 1.0, 10.0]
        }
    },

    # Lasso regression used as feature selector (different from LogisticRegression L1 above)
    # alpha controls sparsity: higher alpha = fewer features kept
    'LASSO-Reg': {
        'steps': [
            ('scaler',   RobustScaler()),
            ('selector', SelectFromModel(Lasso()))
        ],
        'extra_params': {
            'selector__estimator__alpha':     [0.1, 1.0]              if DEBUG else [0.01, 0.1, 1.0, 10.0],
            'selector__estimator__max_iter':  [1000]                  if DEBUG else [1000, 3000, 5000],
            'selector__estimator__warm_start':[False]                 if DEBUG else [True, False],
            'selector__estimator__positive':  [True],
            'selector__estimator__selection': ['cyclic']              if DEBUG else ['random', 'cyclic']
        }
    },

    'SelectKBest': {
        'steps': [
            ('scaler',   RobustScaler()),
            ('selector', SelectKBest(score_func=f_classif, k=best_k))  # best_k from section 4
        ],
        'extra_params': {}  # k is already fixed to best_k found above
    }
}

#%%
# 7. Main loop — nested CV for every (selector × classifier) combination
# =============================================================================

# In DEBUG mode: only run the fastest combination (PCA + SVM-RBF) to verify
# the pipeline works end-to-end before committing to the full run.
if DEBUG:
    active_selectors    = {k: feature_selectors[k] for k in ['PCA']}
    active_classifiers  = {k: classifiers[k]        for k in ['SVM-RBF']}
    print("DEBUG: running 1 combination only (PCA + SVM-RBF)")
else:
    active_selectors   = {k: feature_selectors[k] for k in [ 'SelectKBest']}
    active_classifiers = {k: classifiers[k]        for k in ['SVM-RBF']}
    print("SelectKbest + SVM-RBF)")

results = {}   # stores nested CV scores
grids   = {}   # stores fitted GridSearchCV objects for the final models

print("\n" + "=" * 65)
print("  Running nested cross-validation for all combinations …")
print("=" * 65)

for sel_name, sel_info in active_selectors.items():
    for clf_name, clf_info in active_classifiers.items():

        combo = f"{sel_name} + {clf_name}"
        print(f"\n--- {combo} ---")

        # Build pipeline
        # Random Forest doesn't need scaling, but keeping it in doesn't hurt
        pipeline = Pipeline(sel_info['steps'] + [('clf', clf_info['clf'])])

        # Merge selector params and classifier params
        param_grid = {**sel_info['extra_params'], **clf_info['params']}

        # Inner grid search
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=inner_cv,
            n_jobs=-1,
            verbose=0
        )

        # Outer nested CV
        nested_scores = cross_validate(
            estimator=grid_search,
            X=X_train,
            y=y_train,
            cv=outer_cv,
            scoring='roc_auc',
            n_jobs=-1,
            return_estimator=True,
            return_train_score=False
        )

        mean_auc = nested_scores['test_score'].mean()
        std_auc  = nested_scores['test_score'].std()

        results[combo] = {
            'mean_auc': mean_auc,
            'std_auc':  std_auc,
            'fold_scores': nested_scores['test_score']
        }

        print(f"  Fold scores : {np.round(nested_scores['test_score'], 4)}")
        print(f"  Mean AUC    : {mean_auc:.4f}  ±  {std_auc:.4f}")

        # Best params per outer fold (like your old code)
        print(f"  Best params per outer fold:")
        for i, est in enumerate(nested_scores['estimator'], 1):
            print(f"    Fold {i}: {est.best_params_}")

        # Store the grid for later final-model fitting
        grids[combo] = grid_search

#%%
# 8. Summary table
# =============================================================================

print("\n" + "=" * 65)
print("  Nested CV results — ranked by mean ROC-AUC")
print("=" * 65)

summary = pd.DataFrame({
    combo: {'Mean AUC': v['mean_auc'], 'Std AUC': v['std_auc']}
    for combo, v in results.items()
}).T.sort_values('Mean AUC', ascending=False)

print(summary.to_string(float_format='{:.4f}'.format))

#%%
# 9. Bar chart — compare all combinations
# =============================================================================

combos    = list(summary.index)
means     = summary['Mean AUC'].values
stds      = summary['Std AUC'].values
colors    = ['#2196F3' if 'PCA' in c else '#FF7043' for c in combos]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.barh(combos, means, xerr=stds, color=colors,
               capsize=4, edgecolor='white', height=0.6)
ax.set_xlabel('Mean Nested CV ROC-AUC', fontsize=12)
ax.set_title('Feature Selection × Classifier — Nested CV Comparison', fontsize=13)
ax.set_xlim(0.5, 1.0)
ax.axvline(0.8, color='grey', linestyle='--', linewidth=0.8, label='AUC = 0.80')
ax.legend(fontsize=9)

# Value labels
for bar, mean, std in zip(bars, means, stds):
    ax.text(mean + std + 0.005, bar.get_y() + bar.get_height() / 2,
            f'{mean:.3f}', va='center', fontsize=9)

# Colour legend
legend_patches = [Patch(color='#2196F3', label='PCA'),
                  Patch(color='#FF7043', label='LASSO')]
ax.legend(handles=legend_patches, loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'comparison.png'),
            dpi=150)
plt.show()

#%%
# 10. Final model — train best combination on full training set, evaluate on test
# =============================================================================

best_combo = summary.index[0]
print(f"\nBest combination: {best_combo}")
print("Fitting final model on full training set …")

best_grid = grids[best_combo]
best_grid.fit(X_train, y_train)

print(f"\nBest hyperparameters:\n{best_grid.best_params_}")
print(f"Best inner CV ROC-AUC: {best_grid.best_score_:.4f}")

final_model = best_grid.best_estimator_

# --- Test set evaluation (only done once, right here) ---
y_pred       = final_model.predict(X_test)
y_pred_proba = final_model.predict_proba(X_test)[:, 1]
test_auc     = roc_auc_score(y_test, y_pred_proba)

print(f"\n{'=' * 65}")
print(f"  TEST SET RESULTS — {best_combo}")
print(f"{'=' * 65}")
print(f"ROC-AUC: {test_auc:.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm  = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title(f'Confusion Matrix — {best_combo}')
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'confusion_matrix.png'),
            dpi=150)
plt.show()