# Automated ML results (KNN only)
- Job name: knn-automl-Zohaib-hassan-02
- Best algorithm (scaler + model): RobustScaler (quantile range 25-75, no centering) + KNeighborsClassifier
- n_neighbors: 22
- weights: distance
- Distance metric: manhattan
- AUC weighted: 0.925
- Test recall (failure class): 0.144 (40 of 278 failures caught, validation)
- Test F1 (failure class): 0.248 (precision 0.909)
- Observation: My first KNN-only job failed because featurization one-hot encoded the type column, which made the data sparse, and KNN cannot use sparse data. After I converted type to numbers (L=0, M=1, H=2) and set it to Numeric, AutoML trained KNN successfully. AutoML chose RobustScaler with K=22 and distance weighting, while my notebook chose StandardScaler with K=1 and uniform weights (both use manhattan distance). The difference comes from the tuning goal: I optimized F1, and AutoML optimized AUC weighted. A larger K gives smoother probabilities, so AutoML's AUC is higher (0.925 vs 0.669), but at the default 0.5 threshold it catches fewer failures (recall 0.144 vs 0.353) while making very few false alarms (precision 0.909).
