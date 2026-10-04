import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)


def azureml_main(dataframe1=None, dataframe2=None):
    train = dataframe1.copy()
    test = dataframe2.copy()

    TARGET = "machine_failure"
    NUM_COLS = ["air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min"]
    CAT_COLS = ["type"]

    train[NUM_COLS] = train[NUM_COLS].astype(float)
    test[NUM_COLS] = test[NUM_COLS].astype(float)

    X_train = train[NUM_COLS + CAT_COLS]
    y_train = train[TARGET].astype(int)
    X_test = test[NUM_COLS + CAT_COLS]
    y_test = test[TARGET].astype(int)

    prep = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    model = Pipeline([
        ("prep", prep),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    predictions = test.copy()
    predictions["predicted_failure"] = pred
    predictions["failure_probability"] = proba

    metrics = pd.DataFrame({
        "metric": ["accuracy", "precision", "recall", "f1", "auc"],
        "value": [
            accuracy_score(y_test, pred),
            precision_score(y_test, pred, zero_division=0),
            recall_score(y_test, pred),
            f1_score(y_test, pred),
            roc_auc_score(y_test, proba),
        ],
    })
    return predictions, metrics
