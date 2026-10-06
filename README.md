# azureml-knn-maintenance
KNN predictive maintenance on Azure ML (Notebook, AutoML, Designer)
## Results Comparison

| | Notebook | Automated ML (KNN) | Designer |
|---|---|---|---|
| Best K | 1 (p=1) | 22 | 5 (fixed, not tuned) |
| Weights | uniform | distance | uniform |
| Scaler | StandardScaler | RobustScaler | StandardScaler |
| Test recall (failure) | 0.353 | 0.144 | 0.294 |
| Test F1 (failure) | 0.397 | 0.248 | 0.435 |
| AUC | 0.669 | 0.925 | 0.829 |
| Code written | Most | None | Small script |
| Time to set up (min) | 60 | 30 | 30 |
| Deployable to managed endpoint | Yes | Yes | No (classic components) |
| Best for | Full control, learning | Fast search, baseline | Visual teams, quick prototypes |


# Predictive Maintenance with KNN on Azure Machine Learning

**Student:** Zohaib Hassan

## Project Overview

This project implements a **Predictive Maintenance system using K-Nearest Neighbors (KNN)** on **Microsoft Azure Machine Learning**.

The goal is to predict whether a machine is likely to experience a failure based on machine operating conditions such as:

- Air temperature
- Process temperature
- Rotational speed (RPM)
- Torque
- Tool wear
- Machine type

Three different approaches were explored and compared:

1. **KNN Notebook** using scikit-learn and MLflow
2. **Automated ML** restricted to KNN
3. **Azure ML Designer** using a custom Python KNN script

The trained model was also deployed and tested using an **Azure ML managed online endpoint**.

---

## Problem

Predictive maintenance aims to identify machines that are likely to fail before an actual failure occurs.

The dataset contains approximately **3.4% machine failures**, making it an imbalanced classification problem. Because failures are relatively rare, accuracy alone is not a sufficient evaluation metric.

For this project, particular attention was given to:

- Failure recall
- Precision
- F1-score
- AUC
- Confusion matrix

The main objective is to reduce missed machine failures while maintaining reasonable overall model performance.

---

## Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset**.

The original dataset contains 10,000 observations. The data preparation notebook cleans the dataset and creates:

```text
data/ai4i_clean.csv
```

The cleaned dataset contains the following main features:

| Feature           | Description          |
| ----------------- | -------------------- |
| `type`            | Machine/product type |
| `air_temp_k`      | Air temperature      |
| `process_temp_k`  | Process temperature  |
| `rpm`             | Rotational speed     |
| `torque_nm`       | Torque               |
| `tool_wear_min`   | Tool wear            |
| `machine_failure` | Target variable      |

ID and failure-mode columns were removed during preprocessing to avoid leakage.

The failure class represents approximately **3.4%** of the dataset.

---

# Project Approaches

## 1. KNN Notebook — scikit-learn + MLflow

The main KNN model was implemented in:

```text
notebooks/01_knn_notebook.ipynb
```

The notebook includes:

- Data loading
- Exploratory data analysis
- Feature preprocessing
- StandardScaler
- One-hot encoding for machine type
- KNN classification
- Comparison of scaled and unscaled data
- GridSearchCV
- Stratified cross-validation
- Model evaluation
- Confusion matrix
- MLflow experiment tracking
- Model registration

The hyperparameter search considered:

- Number of neighbors (`K`)
- Distance weighting
- Manhattan distance
- Euclidean distance

The best configuration was:

```text
K = 1
Distance = Manhattan (p=1)
Weights = uniform
Scaler = StandardScaler
```

Test performance:

```text
Accuracy  = 0.964
Precision = 0.453
Recall    = 0.353
F1-score  = 0.397
AUC       = 0.669
```

---

## 2. Automated ML — KNN Only

Automated ML results are documented in:

```text
automl/automl_results.md
```

The AutoML experiment was restricted to the KNN algorithm.

The best configuration found by Automated ML was:

```text
K = 22
Weights = distance
Distance = Manhattan
Scaler = RobustScaler
```

The AutoML experiment optimized for weighted AUC.

Recorded results:

```text
AUC             = 0.925
Failure Recall  = 0.144
Failure F1      = 0.248
Precision       = 0.909
```

The first KNN-only AutoML attempt encountered an issue related to sparse one-hot encoding. The machine type was subsequently converted to a numeric representation, after which the AutoML experiment completed successfully.

---

## 3. Azure ML Designer

The Designer implementation uses the custom Python script:

```text
designer/knn_designer_script.py
```

The script:

- Receives training/test data
- Separates features and target
- Applies StandardScaler
- Applies OneHotEncoder to machine type
- Trains a KNN classifier
- Generates predictions
- Generates prediction probabilities
- Calculates evaluation metrics

The Designer KNN configuration used:

```text
K = 5
Weights = uniform
Scaler = StandardScaler
```

Recorded Designer results:

```text
Failure Recall = 0.294
Failure F1     = 0.435
AUC            = 0.829
```

The Designer approach provides a visual workflow and requires less code than the notebook approach.

---

# Results Comparison

|                                | Notebook               | Automated ML (KNN)    | Designer                       |
| ------------------------------ | ---------------------- | --------------------- | ------------------------------ |
| Best K                         | 1 (p=1)                | 22                    | 5 (fixed, not tuned)           |
| Weights                        | uniform                | distance              | uniform                        |
| Scaler                         | StandardScaler         | RobustScaler          | StandardScaler                 |
| Test recall (failure)          | 0.353                  | 0.144                 | 0.294                          |
| Test F1 (failure)              | 0.397                  | 0.248                 | 0.435                          |
| AUC                            | 0.669                  | 0.925                 | 0.829                          |
| Code written                   | Most                   | None                  | Small script                   |
| Time to set up (min)           | 60                     | 30                    | 30                             |
| Deployable to managed endpoint | Yes                    | Yes                   | No (classic components)        |
| Best for                       | Full control, learning | Fast search, baseline | Visual teams, quick prototypes |

---

# Which Approach Gave the Best Recall?

The **Notebook approach** achieved the highest failure recall:

```text
Notebook:      0.353
Designer:      0.294
Automated ML:  0.144
```

Therefore, the Notebook approach was the best approach for detecting actual machine failures in this comparison.

Recall is particularly important in predictive maintenance because failing to detect a real machine failure can result in unexpected downtime, equipment damage, and maintenance costs.

---

# Which Approach Would I Choose for a Real Factory?

For this project, I would choose the **Notebook approach** as the starting point for a real factory deployment.

The main reasons are:

- It provides greater control over preprocessing.
- Hyperparameters can be explicitly tuned.
- The model can be optimized for the desired evaluation metric.
- MLflow can be used to track experiments.
- The model can be deployed to a managed online endpoint.
- It achieved the highest failure recall among the three approaches in this comparison.

However, a final production decision should also consider the cost of false positives and false negatives, model latency, maintenance requirements, monitoring, and the operational requirements of the factory.

---

# Is the 3.4% Failure Rate a Problem?

Yes. The dataset is highly imbalanced because only approximately **3.4% of observations represent machine failures**.

This means accuracy can be misleading.

For example, a model could achieve high accuracy by predicting that most machines will not fail while still missing a large number of actual failures.

Therefore, this project focuses on metrics such as:

- Recall
- Precision
- F1-score
- AUC
- Confusion matrix

Possible improvements for the class imbalance problem include:

- Collecting more failure examples
- Applying class-balancing techniques
- Adjusting the classification threshold
- Optimizing specifically for failure recall
- Testing alternative machine-learning algorithms
- Using additional maintenance and sensor data

---

# Azure ML Deployment

The deployment notebook is:

```text
notebooks/02_deploy_endpoint.ipynb
```

The project uses an **Azure Machine Learning managed online endpoint**.

Deployment configuration:

```text
Endpoint: knn-maint-zohaibhassan2
Deployment: blue
VM Size: Standard_DS2_v2
```

The deployed model was successfully tested and produced predictions such as:

```text
Prediction (0 = OK, 1 = failure): [0, 1]
```

The deployment notebook also includes endpoint deletion/cleanup.

A sample request is available at:

```text
deployment/sample-request.json
```

The endpoint testing script is:

```text
deployment/test_endpoint.py
```

The endpoint key is not stored directly in the repository. The test script reads the endpoint URL and key from environment variables:

```text
AML_ENDPOINT_URL
AML_ENDPOINT_KEY
```

This helps prevent sensitive endpoint credentials from being committed to GitHub.

---

# Repository Structure

```text
azureml-knn-maintenance/
│
├── automl/
│   └── automl_results.md
│
├── data/
│   └── ai4i_clean.csv
│
├── deployment/
│   ├── sample-request.json
│   └── test_endpoint.py
│
├── designer/
│   └── knn_designer_script.py
│
├── notebooks/
│   ├── 00_prepare_data.ipynb
│   ├── 01_knn_notebook.ipynb
│   └── 02_deploy_endpoint.ipynb
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

# Main Files

### `notebooks/00_prepare_data.ipynb`

Prepares and cleans the AI4I predictive maintenance dataset and creates:

```text
data/ai4i_clean.csv
```

It also registers the cleaned data as an Azure ML data asset.

### `notebooks/01_knn_notebook.ipynb`

Contains the main KNN machine-learning workflow, including preprocessing, scaling, hyperparameter tuning, evaluation, and MLflow tracking.

### `notebooks/02_deploy_endpoint.ipynb`

Registers the model and deploys it to an Azure ML managed online endpoint for prediction.

### `automl/automl_results.md`

Contains the results and configuration from the KNN-only Automated ML experiment.

### `designer/knn_designer_script.py`

Contains the custom Python KNN implementation used with Azure ML Designer.

### `deployment/sample-request.json`

Contains a sample input request for the deployed model.

### `deployment/test_endpoint.py`

Sends a request to the deployed endpoint and displays the prediction.

---

# Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- MLflow
- Azure Machine Learning
- Azure ML Designer
- Automated ML
- GitHub
- K-Nearest Neighbors (KNN)

---

# What I Learned

Through this project, I learned how to:

- Prepare and clean a predictive-maintenance dataset.
- Handle imbalanced classification data.
- Apply feature scaling and categorical encoding.
- Train and tune a KNN classifier.
- Use GridSearchCV and cross-validation.
- Evaluate models using recall, precision, F1-score, and AUC.
- Track experiments using MLflow.
- Run KNN using Azure Automated ML.
- Create a custom Python component for Azure ML Designer.
- Register models in Azure Machine Learning.
- Deploy a model using a managed online endpoint.
- Send prediction requests to a deployed model.
- Compare different machine-learning workflows.
- Manage machine-learning project files using GitHub.

---

# Conclusion

This project demonstrates three different ways to build a KNN predictive-maintenance solution using Azure Machine Learning.

The **Notebook approach achieved the highest failure recall (0.353)**, making it the strongest approach in this comparison when the priority is detecting machine failures.

The **Automated ML approach achieved the highest AUC (0.925)**, while the **Designer approach achieved the highest F1-score (0.435)** among the three approaches.

Overall, the project demonstrates the trade-offs between manual model development, automated model search, and visual machine-learning workflows in Azure Machine Learning.
