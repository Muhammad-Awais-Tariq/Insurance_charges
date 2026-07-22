# Insurance Charges Prediction

An end-to-end machine learning project predicting individual medical insurance charges, built with a full preprocessing pipeline, compared across four different models, and deployed as an interactive Streamlit web app.

## Live Demo

**Try it here:** [https://insurancecharges12.streamlit.app/](https://insurancecharges12.streamlit.app/)

Enter a person's details (age, sex, BMI, children, smoker status, region) and get a live insurance charge prediction from the trained Random Forest model.

---

## Project Structure

```
Insurance_Charges/
│
├── Data/                       # Raw dataset (gitignored — not pushed to GitHub)
│   └── insurance.csv           # Original Kaggle insurance dataset
│
├── exploration.ipynb           # Notebook: EDA, feature engineering iteration,
│                                # model comparison, hyperparameter tuning
├── final_pipeline.py           # Clean, final training script — builds the
│                                # preprocessing pipeline, trains the final
│                                # model, and saves it with joblib
├── app.py                      # Streamlit app — loads the saved model and
│                                # serves the interactive prediction UI
├── tree_pipeline.joblib        # Serialized final trained pipeline (model +
│                                # preprocessing, all in one)
│
├── pyproject.toml              # Project dependencies (for uv)
├── uv.lock                     # Locked dependency versions
├── .python-version             # Python version pin
├── .gitignore                  # Excludes Data/ folder from version control
└── README.md                   # This file
```

> **Note on data:** The `Data/` folder (raw CSV) is excluded via `.gitignore` and is **not** pushed to GitHub. To run this project locally, download `insurance.csv` from the [Kaggle Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance) and place it in a local `Data/` folder.

---

## What Each File Does

### `exploration.ipynb`
The working notebook where all the experimentation happened:
- Exploratory data analysis on the raw insurance dataset
- Preprocessing and feature exploration (age, sex, BMI, children, smoker, region)
- Building and debugging the sklearn `Pipeline` + `ColumnTransformer`
- Training and cross-validating four different models
- Hyperparameter tuning with `RandomizedSearchCV`
- Comparing RMSE across all models

### `final_pipeline.py`
The clean, production version of the pipeline:
1. Loads `insurance.csv`, splits into train/test
2. Builds the full `Pipeline`: preprocessing (scaling, encoding) → model
3. Fits the final chosen model (Random Forest)
4. Evaluates RMSE on the held-out test set
5. Serializes the trained pipeline with `joblib` for deployment

### `app.py`
The Streamlit web app. Loads `tree_pipeline.joblib`, presents a form for entering personal details, and returns a live insurance charge prediction.

---

## Preprocessing

- **Numeric columns** (`age`, `bmi`, `children`) → `StandardScaler`
- **Categorical columns** (`sex`, `smoker`, `region`) → `OneHotEncoder`

All preprocessing steps are fit once on the training set and reused at inference time, ensuring the pipeline works correctly on a single new input, not just batches.

---

## Models Compared

Four models were trained and evaluated using RMSE:

| Model | RMSE |
|---|---|
| Linear Regression | 5828.22 |
| Decision Tree | 6582.06 |
| Random Forest | 4908.00 |
| **Random Forest (tuned, final)** | **4427.63** |

**Winner: Random Forest** — best performer after hyperparameter tuning, and was selected as the final deployed model.

### Notes on the results
- Hyperparameter tuning improved the Random Forest's RMSE from 4908.00 to 4427.63, the largest gain of any model in this comparison.
- The Decision Tree performed worse than Linear Regression here, suggesting a single tree overfits or underfits parts of the data that ensembling in Random Forest handles better.

---

## How to Run Locally

### Prerequisites
- Python (see `.python-version`)
- `uv` package manager (or `pip`)

### Setup
```bash
git clone <repository-url>
cd Insurance_Charges

# Download insurance.csv from Kaggle's Medical Cost Personal Datasets
# and place it inside a local Data/ folder (not included in this repo)

uv sync
```

### Train the model
```bash
uv run python final_pipeline.py
```
This regenerates `tree_pipeline.joblib`.

### Run the Streamlit app
```bash
uv run streamlit run app.py
```
Then open the URL shown in your terminal (usually `http://localhost:8501`).

---

## Technologies Used

- **[scikit-learn](https://scikit-learn.org/)** — pipelines, preprocessing, models, cross-validation, hyperparameter search
- **[Pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/)** — data manipulation
- **[Streamlit](https://streamlit.io/)** — interactive web app deployment
- **[joblib](https://joblib.readthedocs.io/)** — model serialization

---

## Key Learnings from This Project

- Building a leak-free sklearn `Pipeline`/`ColumnTransformer` from raw data to model
- Comparing linear, tree-based, and ensemble models using RMSE
- Diagnosing the impact of hyperparameter tuning on model error
- Serializing a full pipeline (preprocessing + model) for deployment
- Deploying a trained pipeline behind a live Streamlit interface

---

## Author

Muhammad Awais Tariq

If you found this project useful, consider giving it a star.