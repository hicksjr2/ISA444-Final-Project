# Business Forecasting Using Statistical, ML, Deep Learning and Transformer Models
## By John Hicks, Caleb Vowell, & Mia Weber

## Dataset
In this class project, we utilize the [Walmart Sales Forecasting dataset from Kaggle](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast) to compare a wide range of time-series forecasting models. 
Given that the full Walmart dataset contains hundreds of Store-Department series, we focused on the top 40 highest-volume time series. We selected these series because they have dense weekly histories, strong seasonal patterns, and stable sales behavior - making them ideal for comparing forecasting models.

## Forecasting Procedure
Our approach consisted of the following steps:  
1. Loaded, cleaned, and downsampled the Walmart dataset by selecting the top 40 Store–Dept series ranked by total sales volume.
2. Reformatted the resulting dataset into Nixtla’s three-column panel structure (unique_id, ds, y).
3. Saved the processed dataset as downsampled_df.csv, creating a consistent input across all forecasting models.
4. Applied 5-fold backtesting, where each fold trains on historical data and predicts the next 13 weeks for all 40 series.
5. Computed ME, MAE, RMSE, and MAPE for every model, fold, and series to quantify forecasting accuracy.
6. Identified per-series model winners and found that Seasonal Naive outperformed Naive on ~80% of series across multiple metrics.
7. Examined forecast plots to assess how Seasonal Naive captured recurring seasonal effects and holiday peaks more effectively than Naive.
8. Trained classical statistical models (AutoARIMA, AutoETS, Naive, Seasonal Naive) using StatsForecast and compared their multi-fold CV performance.
9. Implemented a LightGBM machine learning model via MLForecast using lag and calendar features and evaluated performance against statistical baselines.
10. Trained deep learning models (AutoNBEATS and AutoNHITS) and evaluated them using cross-validation with tuned hyperparameters.
11. Applied transformer-based foundation models (Chronos, Moirai, TimesFM-2.0, TimesFM-2.5, and TimeCopilot statistical models) and evaluated performance using MASE.
12. Synthesized results across statistical, ML, deep learning, and foundation models to understand forecasting behavior and benchmark model performance.

## Results and Observations
Our evaluation compared multiple forecasting approached across the top 40 Store-Dept time series. We will reports 1) ME, MAE, RMSE, and MAPE for all methods and 2) Model-winner counts. Below is a concise summmary of findings.

1. Baseline Model Performance (Naive vs. Seasonal Naive)
- Seasonal Naive wins majority of series for ME, MAE, RMSE, and MAPE
  * ME: 31 wins vs. 9 wins
  * MAE: 28 wins vs. 12 wins
  * RMSE: 31 wins vs. 9 wins
  * MAPE: 30 wins vs. 10 wins
- This confirms that weekly Walmart sales are highly seasonal. Incorporating last-year seasonality provides a strong baseline for our forecasting exploration.
- These metrics and winner counts are visible the following CSV files:
  * [per_series_model_metrics.csv](https://github.com/hicksjr2/ISA444-Final-Project/blob/main/per_series_model_metrics.csv)
  * [winner_counts_ME.csv](https://github.com/hicksjr2/ISA444-Final-Project/blob/main/winner_counts_ME.csv), [winner_counts_MAE.csv](https://github.com/hicksjr2/ISA444-Final-Project/blob/main/winner_counts_MAE.csv), [winner_counts_RMSE.csv](https://github.com/hicksjr2/ISA444-Final-Project/blob/main/winner_counts_RMSE.csv), [winner_counts_MAPE.csv](https://github.com/hicksjr2/ISA444-Final-Project/blob/main/winner_counts_MAPE.csv)

2. Classical Statistical Models (StatsForecast)
- 

2. Machine Learning (LightGBM)
- 

3. Deep Learning (AutoNBEATS & AutoNHITS)
- 

4. Foundation Models (TimeCoPilot)
- 

### Summary of Findings:
- Seasonal Naive is the strongest simple baseline, winning most series for ME, MAE, RMSE, and MAPE.
- AutoETS and AutoARIMA improve accuracy over naive models, especially on stable weekly series.
- LightGBM performs well on autoregressive patterns but is sensitive to volatility.
- Deep learning models (especially NHITS) provide large improvements for complex seasonal structure.
- Foundation models (TimesFM-2.5 & TimesFM-2.0) are the top performers overall, achieving the lowest MASE across all models.

## Our Python Notebooks
The following notebooks contain our code, results and insights:  
  - [Downsampling Process](https://github.com/hicksjr2/ISA444-Final-Project/blob/main/ISA444_downsample_preprocess.ipynb)

Contains data cleaning, selection of the top 40 Store-Dept series, and creation of the Nixtla-formatted panel dataset ('downsampled_df.csv').
  - [Forecasting Models Colab](https://colab.research.google.com/drive/14YOiFIOjZzcY80prRuOus06nuGeIG7AE?usp=sharing) *Use Google Colab to view code*

Includes all forecasting models (baseline, statistical, ML, deep learning, and foundation models), cross-validation, metrics, and evaluation.

