

# Model Card — CancerGuard SVM Breast Cancer Classifier

## 1. Model Overview

**CancerGuard** is a binary machine learning classifier developed to predict whether a breast tumour is **malignant or benign** using numerical diagnostic tumour measurements.

The model uses a **Support Vector Machine (SVM)** implemented with `scikit-learn` and is deployed through a FastAPI REST API for real-time inference.

| Attribute           | Description                              |
| ------------------- | ---------------------------------------- |
| Model Name          | CancerGuard SVM Breast Cancer Classifier |
| Model Type          | Binary Classification                    |
| Algorithm           | Support Vector Machine (SVM)             |
| Framework           | scikit-learn                             |
| Input               | 30 numerical diagnostic features         |
| Output              | Cancer`present` or `absent`          |
| Deployment          | FastAPI REST API                         |
| Model Artifact      | `models/final_svm_pipeline.pkl`        |
| Operating Threshold | `0.30`                                 |
| Status              | Experimental / Educational Prototype     |

---

## 2. Intended Use

CancerGuard was developed for:

- Demonstrating an end-to-end machine learning classification workflow
- Exploring SVM-based breast cancer classification
- Demonstrating model preprocessing, training, evaluation, and serialization
- Demonstrating deployment of a trained ML model through FastAPI
- Educational and research experimentation
- API development and automated ML system testing

### Out-of-Scope Uses

The model is **not intended for**:

- Clinical diagnosis
- Patient screening
- Treatment recommendations
- Clinical decision-making
- Replacement of qualified healthcare professionals
- Deployment as a regulated medical device

---

## 3. Dataset

The model was developed using the **Wisconsin Breast Cancer Diagnostic dataset**.

| Dataset Property              | Value                   |
| ----------------------------- | ----------------------- |
| Number of Observations        | 569                     |
| Number of Predictive Features | 30                      |
| Problem Type                  | Binary Classification   |
| Target                        | Breast tumour diagnosis |
| Malignant Label               | `M`                   |
| Benign Label                  | `B`                   |

During model development, the target was encoded as:

```text
Malignant (M) = 1
Benign (B)    = 0
```

Non-predictive identifiers and empty CSV artifacts were excluded from model training.

---

## 4. Input Features

CancerGuard uses 30 numerical features describing characteristics of cell nuclei derived from breast mass measurements.

The features are organised into three groups: **mean measurements, standard-error measurements, and worst measurements**.

### Mean Features

- `radius_mean`
- `texture_mean`
- `perimeter_mean`
- `area_mean`
- `smoothness_mean`
- `compactness_mean`
- `concavity_mean`
- `concave_points_mean`
- `symmetry_mean`
- `fractal_dimension_mean`

### Standard Error Features

- `radius_se`
- `texture_se`
- `perimeter_se`
- `area_se`
- `smoothness_se`
- `compactness_se`
- `concavity_se`
- `concave_points_se`
- `symmetry_se`
- `fractal_dimension_se`

### Worst Features

- `radius_worst`
- `texture_worst`
- `perimeter_worst`
- `area_worst`
- `smoothness_worst`
- `compactness_worst`
- `concavity_worst`
- `concave_points_worst`
- `symmetry_worst`
- `fractal_dimension_worst`

All 30 features are required when submitting a prediction request to the CancerGuard API.

---

## 5. Data Preprocessing

The model-development workflow included:

1. Removal of non-predictive identifier fields
2. Removal of empty CSV artifacts
3. Validation of missing values and dataset quality
4. Binary target encoding
5. Stratified train/test splitting
6. Feature scaling
7. Model training using the training partition
8. Evaluation using an independent held-out test set

Feature preprocessing is incorporated into the serialized model pipeline used by the API:

```text
Input Features
      ↓
Preprocessing / Scaling
      ↓
SVM Classifier
      ↓
Probability Estimate
      ↓
0.30 Operating Threshold
      ↓
present / absent
```

The final serialized pipeline is stored at:

```text
models/final_svm_pipeline.pkl
```

---

## 6. Model Development

Multiple SVM configurations were evaluated during model development.

The development process included:

- SVM baseline modelling
- Kernel comparison
- Hyperparameter optimisation
- Cross-validation
- Evaluation using classification metrics
- Probability-based threshold analysis
- Final operating-threshold selection

### Final Model Configuration

| Parameter             | Selected Value               |
| --------------------- | ---------------------------- |
| Algorithm             | Support Vector Machine (SVM) |
| Preprocessing         | `StandardScaler`           |
| Kernel                | `rbf`                      |
| `C`                 | `10`                       |
| `gamma`             | `0.01`                     |
| `class_weight`      | `balanced`                 |
| Probability Estimates | `True`                     |
| `random_state`      | `42`                       |
| Operating Threshold   | `0.30`                     |

The final model is implemented as a scikit-learn `Pipeline` combining `StandardScaler` with an RBF-kernel SVM classifier.

**Kernel and hyperparameter selection were supported during model development and optimisation. The production inference endpoint uses the locked final model configuration to ensure consistent predictions.**

---

## 7. Operating Threshold

The final classification threshold was selected using **out-of-fold (OOF) probability estimates generated exclusively from the training set**.

This approach prevents information from the independent test set from influencing threshold optimisation.

A minimum malignant sensitivity criterion of **98%** was applied to the OOF training predictions.

Among candidate thresholds satisfying this criterion, a probability threshold of:

```text
0.30
```

achieved the highest specificity and was selected as the experimental operating threshold.

The API therefore classifies a sample as:

```text
P(Malignant) >= 0.30  →  Cancer Present
P(Malignant) <  0.30  →  Cancer Absent
```

The threshold is explicitly applied during API inference.

---

## 8. Model Performance

The final model was evaluated on the independent held-out test set using the selected `0.30` operating threshold.

| Metric               |      Performance |
| -------------------- | ---------------: |
| Accuracy             | **98.25%** |
| Precision (PPV)      | **97.62%** |
| Sensitivity (Recall) | **97.62%** |
| Specificity          | **98.61%** |
| F1 Score             | **97.62%** |
| False Positives      |      **1** |
| False Negatives      |      **1** |

### Threshold Comparison

Compared with a conventional probability threshold of `0.50`, the selected `0.30` threshold reduced false negatives:

```text
0.50 threshold → 2 False Negatives
0.30 threshold → 1 False Negative
```

while maintaining the same overall accuracy, at the cost of one additional false-positive prediction.

The lower operating threshold was retained to place greater emphasis on identifying malignant cases.

---

## 9. Model Output

The deployed API converts the binary model output into human-readable labels:

| Model Class | API Output  |
| ----------- | ----------- |
| `1`       | `present` |
| `0`       | `absent`  |

Example API response:

```json
{
  "prediction": "present",
  "confidence": 0.9999809919981356
}
```

The `confidence` field represents the model probability associated with the predicted class.

---

## 10. Validation & Testing

The deployed model and API are covered by automated tests using `pytest`.

The test suite validates:

- API health
- Valid prediction requests
- Missing input-field validation
- Missing API-key handling
- Invalid API-key handling
- Prediction latency
- Serialized model loading
- Model inference

The final test suite result was:

```text
8/8 tests passed
```

A manual Swagger UI smoke test was also performed after integrating the `0.30` operating threshold, returning a successful HTTP `200` prediction response.

---

## 11. Model Explainability

Model explainability analysis was performed separately as part of the machine learning development workflow.

The project includes:

```text
notebook/06_Model_Explainability.ipynb
```

This notebook investigates model behaviour and feature contributions to improve understanding of the classifier's predictions.

Explainability results should be interpreted as analytical aids rather than clinical explanations.

---

## 12. Limitations

The model has several important limitations:

- It was developed using a relatively small dataset of 569 observations.
- Evaluation was performed using an internal held-out test set.
- The reported performance has not been externally or clinically validated.
- Performance may change when applied to populations or measurement procedures different from the development dataset.
- The model requires all expected numerical diagnostic features.
- Predictions depend on the quality and consistency of the supplied measurements.
- Dataset shift may reduce model performance.
- Probability estimates should not be interpreted as definitive clinical probabilities.
- The `0.30` operating threshold is experimental and not a medically established diagnostic cutoff.
- The current API is a prototype and does not implement the complete governance, privacy, security, monitoring, and regulatory controls required for a production healthcare system.

---

## 13. Ethical & Clinical Considerations

False-negative predictions are particularly important in cancer classification because they represent malignant cases incorrectly classified as benign.

For this reason, threshold selection placed additional emphasis on malignant sensitivity.

However, changing the operating threshold introduces a trade-off between false negatives and false positives. The selected threshold therefore represents an **experimental modelling decision**, not a clinical recommendation.

Performance should also be assessed across representative populations before any consideration of real-world healthcare use.

---

## 14. Medical Disclaimer

> **CancerGuard is not a medical device and must not be used for clinical diagnosis, patient screening, treatment decisions, or as a replacement for professional medical judgement.**

The model was created for **educational, research, portfolio, and software demonstration purposes**.

Neither the model nor its `0.30` operating threshold has undergone the external validation, prospective clinical evaluation, regulatory review, or medical-device assessment necessary for clinical deployment.

---

## 15. Model Version & Maintenance

| Property            | Value                        |
| ------------------- | ---------------------------- |
| Model Version       | `1.0`                      |
| Model Artifact      | `final_svm_pipeline.pkl`   |
| API                 | CancerGuard FastAPI          |
| Current Status      | Experimental Prototype       |
| Validation          | Internal test-set evaluation |
| External Validation | Not performed                |
| Clinical Validation | Not performed                |

Any future model update should be accompanied by:

- Model re-evaluation
- Threshold re-evaluation
- Regression testing
- Updated performance metrics
- Updated model documentation
- Versioned model artifacts
