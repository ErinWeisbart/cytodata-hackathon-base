try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except Exception as ex:
    print(f'Unable to use intelex version: {str(ex)}')

import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Config libs
# Adapt font size of plots
plt.rcParams.update({'font.size': 18})
import plotly.io as pio
pio.renderers.default = "png" # Set to "svg" or "png" for static plots or "notebook_connected" for interactive plots

from dataset import load_data
from grid_search import rf_search, svc_search
from evaluate import eval_test_set


criteria = 'f1-score'
agg_metric = 'macro avg'

# df_mitocells = pd.read_csv('../data/mitocells.csv')
df_mitocells = pd.read_csv('../data/all_cells_with_features.csv')
selected_features = ['nuclear_volume', 'nuclear_height', 'nuclear_surface_area',
                        'cell_volume', 'cell_height', 'cell_surface_area',
                        ]
for col in df_mitocells:
    if col.startswith('Shape') or col.startswith('Granularity'):
        selected_features.append(col)

selected_features = list(df_mitocells[selected_features].isna().any()[lambda x:~x].index)
X_train, y_train, X_val, y_val, X_test, y_test = load_data(df_mitocells, selected_features)


# Random Forest Baseline

best_rf, df, fig_rf = rf_search(X_train, y_train, X_val, y_val, criteria, agg_metric)
df_val = df.query('split=="valid"').sort_values(by=criteria, ascending=False)
best_rf_conf = list(df_val[['max_depth', 'n_estimators']].iloc[0].values)
cm_fig_rf, y_trainpred_rf, y_testpred_rf = eval_test_set(
        df_mitocells, best_rf, X_train, y_train, X_test, y_test)

print(classification_report(y_test, y_testpred_rf))

# SVC Baseline
best_svc, df, fig_svc = svc_search(X_train, y_train, X_val, y_val, criteria, agg_metric)
df_val = df.query('split=="valid"').sort_values(by=criteria, ascending=False)
best_svc_conf = list(df_val[['gamma', 'C']].iloc[0].values)
cm_fig_svc, y_trainpred_svc, y_testpred_svc = eval_test_set(
        df_mitocells, best_svc, X_train, y_train, X_test, y_test)

print(classification_report(y_test, y_testpred_svc))
