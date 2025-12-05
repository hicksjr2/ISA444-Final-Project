# Business Forecasting Using Statistical, ML, Deep Learning and Transformer Models

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
ABCD


## Our Python Notebooks
The following notebooks contain our code, results and insights:  
  - [Downsampling Process](https://github.com/hicksjr2/ISA444-Final-Project/blob/main/ISA444_downsample_preprocess.ipynb)

Contains data cleaning, selection of the top 40 Store-Dept series, and creation of the Nixtla-formatted panel dataset ('downsampled_df.csv').
  - [Forecasting Models Colab](https://colab.research.google.com/drive/14YOiFIOjZzcY80prRuOus06nuGeIG7AE?usp=sharing) *Use Google Colab to view code*

Includes all forecasting models (baseline, statistical, ML, deep learning, and foundation models), cross-validation, metrics, and evaluation.

