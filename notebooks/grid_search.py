from tqdm.auto import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pandas as pd
import plotly.graph_objects as go
import numpy as np


def rf_search(X_train, y_train, X_val, y_val, criteria, agg_metric):
    max_depth_opts = [2, 4, 8, 16]
    n_estimators_opts = [128, 256, 512, 1024]
    grid = []
    best_model = None
    best_score = -1
    for max_depth in tqdm(max_depth_opts):
        for n_estimators in tqdm(n_estimators_opts, leave=False):
            clf = RandomForestClassifier(max_depth=max_depth,
                                         n_estimators=n_estimators,
                                         random_state=0, class_weight='balanced')
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_train)
            scores = classification_report(
                y_train, y_pred, output_dict=True)[agg_metric]
            scores['max_depth'] = max_depth
            scores['n_estimators'] = n_estimators
            scores['split'] = 'train'
            grid.append(scores)

            y_pred = clf.predict(X_val)
            scores = classification_report(
                y_val, y_pred, output_dict=True)[agg_metric]
            scores['max_depth'] = max_depth
            scores['n_estimators'] = n_estimators
            scores['split'] = 'valid'
            grid.append(scores)

            if best_score < scores[criteria]:
                best_score = scores[criteria]
                best_model = clf
    df = pd.DataFrame(grid)
    z = df.query('split=="valid"')[criteria].values.reshape(
        len(max_depth_opts), len(n_estimators_opts))
    fig = create_fig(
        z=z,
        x=max_depth_opts,  # horizontal axis
        y=n_estimators_opts,  # vertical axis
        title="RF grid search", xaxis_title="max_depth", yaxis_title="n_estimators")
    return best_model, df, fig


def create_fig(z, x, y, title, xaxis_title, yaxis_title):

    fig = go.Figure(data=go.Contour(
        z=z,
        x=x,  # horizontal axis
        y=y  # vertical axis
    ))
    fig.update_layout(
        title="RF grid search",
        xaxis_title="max_depth",
        yaxis_title="n_estimators"
    )
    return fig


def svc_search(X_train, y_train, X_val, y_val, criteria, agg_metric):
    C_opts = np.logspace(-2, 3, 6)
    gamma_opts = np.logspace(-9, 3, 13)
    grid = []
    best_model = None
    best_score = -1

    for gamma in tqdm(gamma_opts):
        for C in tqdm(C_opts, leave=False):
            clf = Pipeline([('scaler', StandardScaler()),
                           ('svc', SVC(gamma=gamma, C=C))])
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_train)
            scores = classification_report(
                y_train, y_pred, output_dict=True, zero_division=0)[agg_metric]

            scores['gamma'] = gamma
            scores['C'] = C
            scores['split'] = 'train'
            grid.append(scores)

            y_pred = clf.predict(X_val)
            scores = classification_report(
                y_val, y_pred, output_dict=True, zero_division=0)[agg_metric]
            scores['gamma'] = gamma
            scores['C'] = C
            scores['split'] = 'valid'
            grid.append(scores)

            if best_score < scores[criteria]:
                best_score = scores[criteria]
                best_model = clf
    df = pd.DataFrame(grid)

    z = df.query('split=="valid"')[criteria].values.reshape(
        len(gamma_opts), len(C_opts))
    fig = create_fig(
        z=z,
        x=gamma_opts,  # horizontal axis
        y=C_opts,  # vertical axis
        title="SVM-RBF grid search", xaxis_title="gamma", yaxis_title="C")

    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    return best_model, df, fig
