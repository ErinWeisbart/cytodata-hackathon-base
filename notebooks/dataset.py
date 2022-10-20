def load_data(df_mitocells, selected_features):

    Xy_train = df_mitocells[df_mitocells['split'] == 'train'][[
        *selected_features, 'cell_stage']].dropna(axis=0).to_numpy()
    X_train = Xy_train[:, 0:-1]
    y_train = Xy_train[:, -1]

    Xy_val = df_mitocells[df_mitocells['split'] == 'valid'][[
        *selected_features, 'cell_stage']].dropna(axis=0).to_numpy()
    X_val = Xy_val[:, 0:-1]
    y_val = Xy_val[:, -1]

    Xy_test = df_mitocells[df_mitocells['split'] == 'test'][[
        *selected_features, 'cell_stage']].dropna(axis=0).to_numpy()
    X_test = Xy_test[:, 0:-1]
    y_test = Xy_test[:, -1]

    return X_train, y_train, X_val, y_val, X_test, y_test
