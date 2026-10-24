# Athlete Fatigue Monitoring and Prediction

A session-by-session web app that predicts an athlete's fatigue class (**Low / Moderate / High**, with a probability percentage for each class) from seven values entered after a training session. Three models (Logistic Regression, Random Forest, XGBoost) are trained in separate scripts, compared on the same held-out data, and **only the selected best model** is used by the app.

---

## 1. How it works

1. **Training stage (offline).** The dataset zip is read, only seven features plus the target are kept, and fatigue classes are created. Three models are trained, each in its own script, and each saves its model, metrics and wall time.
2. **Selection stage.** `training/05_select_best_model.py` compares the three models and writes `models/selected_model.json`.
3. **App stage.** The Flask app loads only the selected model. After each session the athlete enters seven values and gets a class and percentages.
4. **Session 15 onward.** Sessions 1-14 form the initial history window (for demo use they are labelled dummy data). Predictions start at session 15. The comparison with the previous 7 and 14 sessions is context only and is not a model input.

The system is session-by-session. It is not continuous sensor monitoring and not calendar based.

## 2. Model inputs (exactly seven)

`training_duration`, `training_intensity`, `training_load`, `heart_rate`, `sleep_quality`, `recovery_score`, `stress_level`

No other column from the dataset is used (no ids, dates, age, gender, sport, injury columns). `fatigue_index` is the target only and is never an input.

**Classes:** Low if fatigue index < 47.26 (33.63 percentile), Moderate if 47.26 to < 68.89 (33.49 percentile), High if >= 68.89 (32.88 percentile).

## 3. Folder structure

```text
athlete-fatigue-app/
├── README.md
├── requirements.txt
├── .env.example
├── config.py
├── data/
│   ├── raw/                 # copied zip + extracted csv (never edited)
│   └── processed/           # fatigue_7features.csv, train.csv, test.csv
├── training/
│   ├── common.py
│   ├── 00_inspect_dataset.py
│   ├── 01_prepare_data.py
│   ├── 02_train_logistic_regression.py
│   ├── 03_train_random_forest.py
│   ├── 04_train_xgboost.py
│   ├── 05_select_best_model.py
│   └── run_all.py
├── models/
│   ├── logistic_regression/ # model, config, metrics.csv, wall_time.csv, best_params.json
│   ├── random_forest/
│   ├── xgboost/
│   ├── class_config.json
│   └── selected_model.json
├── reports/
│   ├── model_comparison.csv
│   ├── selection_report.md
│   ├── dataset_inspection.txt
│   └── figures/
├── app/
│   ├── app.py
│   ├── models_db.py
│   ├── ml/        (loader.py, features.py, predict.py, labels.py)
│   ├── assistant/ (context.py, llm.py)
│   ├── templates/
│   └── static/
├── seed_demo.py
└── tests/
```

## 4. Setup (Windows)

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then edit .env if configuring optional LLM settings
```

The dataset path is set in `config.py` as `RAW_ZIP_PATH` (default `C:\Users\Resona\Downloads\fatigue datasets\archive (1).zip`) and can be overridden with an environment variable. The original zip is never modified; it is copied to `data/raw/`.

## 5. Run the training pipeline

Run everything in order:

```powershell
python training/run_all.py
```

Or run each stage alone:

```powershell
python training/00_inspect_dataset.py
python training/01_prepare_data.py
python training/02_train_logistic_regression.py
python training/03_train_random_forest.py
python training/04_train_xgboost.py
python training/05_select_best_model.py
```

Each model script saves to `models/<name>/`: the model file, `metrics.csv` (accuracy, precision, recall, F1, ROC-AUC, per-class results), `wall_time.csv` (training and prediction time), `best_params.json`, and plots in `reports/figures/`.

### Training rules

- All three models use the same 7 features and the same train/test split (split method: Stratified random 80/20 train/test split with `random_state=42`).
- Logistic Regression and Random Forest use a median imputer (fitted on training data only inside the pipeline); Logistic Regression also scales features. XGBoost receives missing values as NaN directly.
- Hyperparameter tuning uses 5-fold cross-validation on the training set only. The test set is used once per model for final metrics.
- Target: test accuracy of at least 0.90 for each model, reached only by legitimate means. If a model falls short, the real result is saved, marked `meets_target = false`, and the pipeline records the authentic outcome without falsification.

## 6. Results (filled from the generated files)

| Model | Accuracy | Macro F1 | ROC-AUC | Training time (s) | Meets 0.90 target |
|---|---|---|---|---|---|
| Logistic Regression | 0.6774 (67.74%) | 0.6784 | 0.8523 | 3.55s | No (False) |
| Random Forest | 0.6748 (67.48%) | 0.6773 | 0.8525 | 87.90s | No (False) |
| XGBoost | 0.6699 (66.99%) | 0.6728 | 0.8518 | 214.13s | No (False) |

**Selected model:** `Logistic Regression` (`models/selected_model.json`)

**Selection rule:** Among candidate models, highest test macro-F1 (0.6784); ties within 0.005 broken by ROC-AUC and lower prediction time (0.35ms per 1k samples). Full reasoning is in `reports/selection_report.md`.

## 7. Run the app

```powershell
python seed_demo.py          # creates the demo account with 14 dummy sessions
python app/app.py
```

Open `http://127.0.0.1:5000`. If `models/selected_model.json` is missing, the app shows "Run training/run_all.py first".

**Demo account (local testing only):** `demo@athlete.app` / `Demo@1234`. Change or remove it before any real use. Its next session is session 15.

Dummy sessions 1-14 are labelled "Demo data - illustrative values, not real athlete measurements" wherever they appear.

### Session-15 demo test

Values: duration 70, intensity 8.0, training_load 560, heart_rate 78, sleep_quality 5.5, recovery_score 45, stress_level 0.75.

**Model output (selected model - Logistic Regression):**
- Predicted Class: **Moderate**
- Probabilities: **Moderate 55.9%**, **Low 31.3%**, **High 12.8%**

The history summary for this session (7- and 14-session load sums, means and ratios) is checked by the tests against independently computed values.

## 8. App pages

Dashboard | Add Session | History | Athlete Details | AI Assistant | About the Model | Logout

- **Result page:** predicted class, percentage bar for Low / Moderate / High, workload context (previous 7 and 14 sessions), state labels, "Why this prediction?" factors, and a what-if panel.
- **AI Assistant:** explains results from the athlete's own stored data. It gives no medical advice. With no LLM key configured, starter questions are answered from templates.
- **About the Model:** comparison of the three models, metrics, wall times, plots and limitations, all read from the generated files.

## 9. Run the tests

```powershell
python -m pytest tests/ -v
```

Tests cover: pipeline outputs, no target leakage, saved-model reproducibility, selection rule, the 90% accuracy gate, history summary values, no future data in history windows, no prediction before session 15, missing-value handling, explanations for all three models, authentication and ownership, and UI checks.

## 10. Known limitations

- Accuracy of the selected model is **67.74%** on held-out rows (Macro F1 0.6784); about 32.3 predictions in 100 are wrong.
- The prediction uses only today's seven values. The comparison with previous sessions is context only.
- The most influential inputs are listed from the permutation-importance file; inputs with near-zero importance have little effect on the prediction.
- Sessions are ordered by session number, not calendar date.
- The dataset may be simulated, and Low / Moderate / High are project-defined classes (fatigue index split at about 47.26 and 68.89), not clinical labels.
- Predictions are for monitoring support only. They are not medical advice or a diagnosis.

### Author Documentation & Confirmations

- **Train/Test Split:** Stratified random 80/20 train/test split on the 3 fatigue classes using `random_state=42`.
- **Missing Values:** Logistic Regression and Random Forest utilize a training-set fitted median imputer; XGBoost receives missing values natively as NaN.

## 11. Privacy and security

Passwords are hashed with bcrypt, forms use CSRF tokens, output is escaped, queries are parameterized, and each athlete can see only their own data. Names, emails and passwords are never sent to the LLM. Athletes can delete their account and data from Athlete Details.
