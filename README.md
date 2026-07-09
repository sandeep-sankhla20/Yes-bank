# Yes Bank Stock Closing Price Prediction

A regression capstone project predicting Yes Bank's monthly stock closing price from its Open, High, and Low prices — and studying how the 2018 corporate fraud/NPA scandal reshaped the stock's price behavior.

## 📌 Problem Statement

Predict the monthly closing price of Yes Bank stock using historical Open, High, and Low prices (and engineered features), and analyze the impact of the 2018 fraud case involving the bank's then MD & CEO on the stock's price behavior and predictability.

## 📊 Dataset

- **Source**: Monthly Yes Bank stock price history
- **Period**: July 2005 – November 2020 (185 records)
- **Columns**: `Date`, `Open`, `High`, `Low`, `Close` (all prices in ₹)
- No missing values or duplicates

## 🔍 Project Workflow

1. **Know Your Data** — structure, data types, null/duplicate checks
2. **Understanding Variables** — variable descriptions, summary statistics
3. **Data Wrangling** — OHLC consistency checks, calendar feature extraction
4. **Data Visualization** — 15 charts (univariate, bivariate, multivariate) covering price trends, volatility, correlation heatmap, and pair plots
5. **Hypothesis Testing** — 3 formal tests:
   - Pearson correlation (Open vs. Close): r = 0.978, p < 0.001
   - Welch's t-test (Boom era vs. Crisis era mean Close): p < 0.001
   - One-way ANOVA (Close across 3 eras): p < 0.001
6. **Feature Engineering & Pre-processing** — log transforms, lag/rolling features, cyclical month encoding, VIF-based multicollinearity check, chronological 85/15 train-test split
7. **ML Modeling** — three models trained & hyperparameter-tuned with `TimeSeriesSplit` cross-validation:
   - Linear Regression (baseline)
   - Ridge Regression (`GridSearchCV`)
   - Random Forest Regressor (`RandomizedSearchCV`)
8. **Model Explainability** — Ridge coefficient analysis (feature importance)

## 🏆 Results

| Model | R² | RMSE (₹) | MAPE |
|---|---|---|---|
| **Ridge Regression** ⭐ | 0.966 | 16.64 | 9.62% |
| Linear Regression | 0.963 | 17.38 | 10.12% |
| Random Forest | 0.882 | 30.94 | 23.15% |

**Ridge Regression** was selected as the final model — it best handles the strong multicollinearity between Open/High/Low/lagged Close while achieving the lowest error on a chronologically held-out test set (the most recent, most volatile months).

## 🔑 Key Insight

The model explains ~96.6% of the variance in monthly closing price using only that month's own OHLC data — useful as a fast, explainable estimate. However, no OHLC-only model could have anticipated the 2018 fraud disclosure itself; this is an honest limitation of price-history-only modeling, not a modeling flaw.

## 🛠️ Tech Stack

- Python, pandas, NumPy
- scikit-learn (Linear/Ridge Regression, Random Forest, model selection & tuning)
- statsmodels (VIF, hypothesis testing support)
- scipy.stats (hypothesis testing)
- matplotlib, seaborn (visualization)

## 📁 Repository Contents

- `Sample_ML_Submission_Template.ipynb` — full end-to-end notebook (EDA → hypothesis testing → feature engineering → ML modeling)
- `Sample_EDA_Submission_Template.ipynb` — EDA-only notebook
- `data_YesBank_StockPrices.csv` — dataset
- `Yes_Bank_Insights.pptx` — insights presentation deck
- `Yes_Bank_Video_Script.md` — video narration script

## 🚀 How to Run

1. Clone the repo or download the files:
   ```
   git clone https://github.com/sandeep-sankhla20/Yes-bank.git
   ```
2. Open `Sample_ML_Submission_Template.ipynb` in [Google Colab](https://colab.research.google.com/) or Jupyter
3. Upload `data_YesBank_StockPrices.csv` to the same environment (or your Colab session storage)
4. Run all cells top to bottom

## 🔗 Project Repository

[github.com/sandeep-sankhla20/Yes-bank](https://github.com/sandeep-sankhla20/YesBank-Stock-Prediction)

## ✍️ Author

Sandeep
