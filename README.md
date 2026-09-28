# Predicting Electric Vehicle Purchases

Machine learning project for the **Kaggle Playground Series S6E9 – Predicting Electric Vehicle Purchases** competition.

The objective is to predict the probability that a customer will purchase an electric vehicle. The competition evaluates submissions using **ROC-AUC**.

## Competition

* **Competition:** Playground Series S6E9 – Predicting Electric Vehicle Purchases
* **Task:** Binary classification
* **Evaluation Metric:** ROC-AUC
* **Kaggle:** `anjulaprasad`
* **Final project status:** Baseline completed and submitted

## Dataset

The competition dataset contains customer demographic, financial, transportation, and EV-related features.

| Dataset |    Rows | Columns |
| ------- | ------: | ------: |
| Train   | 668,665 |      15 |
| Test    | 286,571 |      14 |

### Target

`Will_Buy_EV`

| Class |   Count | Proportion |
| ----- | ------: | ---------: |
| No    | 551,886 |     82.54% |
| Yes   | 116,779 |     17.46% |

The target is moderately imbalanced, making probability-based evaluation with ROC-AUC particularly relevant.

### Features

**Numerical**

* `Age`
* `Annual_Income_USD`
* `Daily_Commute_km`
* `Number_of_Cars_Owned`
* `Charging_Stations_Near_Home`
* `Charging_Stations_Near_Work`
* `Environmental_Concern_Level`

**Categorical**

* `Gender`
* `City_Type`
* `Current_Car_Type`
* `Home_Charging_Possible`
* `Subsidy_Available`
* `Range_Anxiety_Level`

The `id` column is unique/sequential and was excluded from model training.

## Project Workflow

```text
Dataset
   ↓
Exploratory Data Analysis
   ↓
Train / Validation Split
   ↓
Feature Preprocessing
   ↓
Logistic Regression Baseline
   ↓
Probability Prediction
   ↓
ROC-AUC Evaluation
   ↓
Kaggle Submission
```

## Exploratory Data Analysis

The EDA focused on understanding the target distribution and identifying relationships between customer characteristics and EV purchase behavior.

Key observations included:

* The target variable is moderately imbalanced.
* `Subsidy_Available` showed a strong relationship with EV purchase rates.
* `Range_Anxiety_Level` showed substantial differences in purchase rates between categories.
* `Environmental_Concern_Level` showed noticeable separation between buyers and non-buyers.
* `Annual_Income_USD` also showed meaningful group-level differences.
* Gender, car ownership, and charging-station features showed comparatively smaller group-level differences.

The complete analysis is available in the competition notebook.

## Preprocessing

The dataset was split into training and validation sets using a stratified 80/20 split.

```python
X = train.drop(columns=["id", "Will_Buy_EV"])
y = train["Will_Buy_EV"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

Numerical features were standardized using `StandardScaler`, while categorical features were encoded using `OneHotEncoder` with `handle_unknown="ignore"`.

The resulting processed feature matrices contained **24 features**.

## Baseline Model

The first competition model was Logistic Regression:

```python
baseline_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

baseline_model.fit(X_train_processed, y_train)
```

Because the competition uses ROC-AUC, predictions were generated as class probabilities rather than hard class labels.

```python
y_valid_proba = baseline_model.predict_proba(
    X_valid_processed
)[:, 1]

baseline_roc_auc = roc_auc_score(
    y_valid,
    y_valid_proba
)
```

## Results

### Local Validation

**ROC-AUC: 0.93796**

### Kaggle

The baseline submission achieved:

**Public ROC-AUC: 0.93737**

| Evaluation          |     ROC-AUC |
| ------------------- | ----------: |
| Local Validation    | **0.93796** |
| Kaggle Public Score | **0.93737** |
| Difference          | **0.00059** |

The small difference between the local validation result and the public leaderboard score indicates that the validation setup produced a reasonably consistent estimate of leaderboard performance for this baseline.

## Kaggle Submission

The submission contains the original test `id` and the predicted probability of purchasing an EV.

```text
submissions/
└── submission_v1_logistic.csv
```

Submission shape:

```text
286,571 × 2
```

The submission was uploaded to Kaggle using:

```bash
kaggle competitions submit \
  -c playground-series-s6e9 \
  -f submissions/submission_v1_logistic.csv \
  -m "Baseline Logistic Regression"
```

## Repository Structure

```text
ev-purchase-prediction/
├── notebooks/
│   └── ev_purchase_prediction.ipynb
├── src/
├── submissions/
│   └── submission_v1_logistic.csv
├── README.md
├── requirements.txt
└── .gitignore
```

## Reproducibility

The project uses a fixed random seed for the train/validation split and model initialization.

Main dependencies include:

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter

Install the project dependencies with:

```bash
pip install -r requirements.txt
```

The complete workflow can be reproduced from:

```text
notebooks/ev_purchase_prediction.ipynb
```

## Project Status

**Completed — Baseline Version**

This project currently represents a completed and reproducible baseline submission.

The implemented workflow covers:

* Dataset exploration
* Target analysis
* Feature identification
* Stratified validation
* Numerical feature scaling
* Categorical feature encoding
* Logistic Regression training
* Probability-based ROC-AUC evaluation
* Kaggle submission generation
* Local vs. leaderboard comparison

Further iterations could investigate feature engineering, alternative tabular models, and targeted hyperparameter tuning.

## Author

**Anjula Prasad**

University of Moratuwa — Information Technology

GitHub: `Anjula2001`

Kaggle: `anjulaprasad`
