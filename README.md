Business Problem
An FMCG brand invests across TV, Digital, Print, Radio, and OOH (Out-of-Home) advertising — but lacks visibility into which channels truly drive revenue and which are overfunded. 

Traditional last-click metrics overvalue bottom-funnel channels and miss the carryover and saturation effects inherent in media spend.

Market Mix Modelling uses regression-based decomposition to attribute revenue back to each channel while accounting for adstock (carryover), diminishing returns (saturation), seasonality, pricing, distribution, promotions, and competitor activity.
---
Pipeline Overview
```
Step 0:  Environment Setup & Synthetic Data Generation
Step 1:  Exploratory Data Analysis (EDA)
Step 2:  Correlation & Multicollinearity Analysis (VIF)
Step 3:  Adstock Transformation (Geometric Decay)
Step 4:  Saturation Curves (Hill Function)
Step 5:  Feature Engineering (Fourier Seasonality, Deviations)
Step 6:  OLS Regression Model + Time-Series Cross-Validation
Step 7:  Model Diagnostics (Residual Analysis, Breusch-Pagan, Durbin-Watson)
Step 8:  Contribution Decomposition & Waterfall Chart
Step 9:  ROI / Effectiveness Analysis (mROI)
Step 10: Budget Optimization via SLSQP (Scenario Simulation)
```
---
Data
Detail	Value
Timeframe	156 weeks (3 years), weekly granularity
Data Type	Synthetic (with known ground-truth coefficients for validation)
Target Variable	Weekly Revenue (₹ Lakhs)
Media Channels	TV, Digital, Print, Radio, OOH
Control Variables	Price Index, Distribution %, Competitor Spend, Promo Depth
> The synthetic data generator encodes realistic marketing dynamics — adstock decay rates, Hill saturation, seasonality, and noise — so we can validate whether the model recovers the true parameters.
---
Key Techniques
Adstock Transformation (Carryover)
Advertising effects don't disappear after one week. Geometric adstock models this carryover:
```
Adstocked(t) = Spend(t) + λ × Adstocked(t-1)
```
Optimal decay rates found via grid search (maximizing correlation with revenue):
TV: 0.70 | Digital: 0.40 | Print: 0.50 | Radio: 0.30 | OOH: 0.55
Hill Saturation (Diminishing Returns)
Doubling spend does NOT double impact. The Hill function captures this:
```
S(x) = x^α / (x^α + K^α)
```
OLS Regression
Chosen for interpretability — explicit coefficients, p-values, confidence intervals. Validated with 5-fold Time-Series Cross-Validation.
Budget Optimization
Given the same total budget, SLSQP optimizer reallocates spend across channels to maximize predicted revenue.
---
Results Summary
Model Performance
Metric	Train	Test
R²	~0.93	~0.89
MAPE	~4.5%	~6.2%
MAE	~3.1	~4.0
Key Findings
Digital & OOH are underfunded relative to their marginal ROI

TV has the highest absolute contribution but lowest marginal returns (saturated)

Budget reallocation could improve predicted weekly revenue by ~8-12% with the same total spend

Visualizations

The notebook contains 15+ publication-quality charts including:

Revenue trends & media spend distributions

Correlation heatmaps & VIF analysis

Adstock decay search curves

Saturation (Hill) curves per channel

Actual vs Predicted (train & test)

4-panel residual diagnostics

Contribution waterfall chart

Spend Share vs Effect Share comparison

ROI & mROI bar charts

Budget optimization comparison
---
Project Structure
```
market-mix-modelling/
│
├── data/
│   ├── raw/                         # Raw data (generated in notebook)
│   └── processed/                   # Transformed features
│
├── notebooks/
│   └── Market_Mix_Modelling_Pipeline.ipynb   # Full 10-step pipeline
│
├── src/
│   ├── adstock.py                   # Adstock transformation functions
│   ├── saturation.py                # Hill saturation curve functions
│   └── optimizer.py                 # Budget optimization via SLSQP
│
├── outputs/                         # Saved charts and result tables
├── requirements.txt
├── README.md
└── LICENSE
Visualization — Matplotlib, Seaborn
