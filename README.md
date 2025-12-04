# Business Forecasting Using Statistical, ML, Deep Learning and Transformer Models

In this class project, we utilize the [Walmart Dataset from Kaggle](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast) to compare various statistical models  for time series. 
Given that the dataset contained XYZ different time-series, we focused on ABC time-series. We picked these timeseries **as they** ... 

## Forecasting Procedure
Our approach consisted of the following steps:  
  1. Loaded, cleaned & downsampled the Walmart dataset to 40 time-series & condensed to a CSV file. We re-shaped the data to match Nixtla's three-column format. The result was a clean weekly time series per store-dept pair. 
  2. We created forecasting models and focused on the naive and seasonal naive methods. These gave a realistic baseline for forecasting and allowed us to test the significance of seasonal patterns. 
  3. We then used backtesting folds: each fold simulates a real forecasting scenario with training, predicting future weeks and comparing against actuals. For each fold, both models generated predictions for the 40 series.
  4. We then computed error metrics for every model, series and fold. We calculated the mean error, mean absolute error, root mean square error, and mean absolute percentage error. The results were combined to determine which model performed best for each store-dept pair.
  5. We identified model winners for each of the 40 series. We compared the naive and seasonal naive for every metric and found that the seasonal naive was the most accurate across ~ 80% of all the series depending on the metric.
  6. We then inspected the forecasts between actual and predicted weekly sales. We discovered that the seasonal naive method captures repetitive annual patterns and peaks during holidays while the naive missed these turning points.
  7. Finally, we combined our findings of the statistical performance measures, visuals, and model winner counts to solidify our insights about the behavior of retail demand and seasonal baselines. 

## Results and Observations
ABCD
![]()

## Our Python Notebooks
The following notebooks contain our code, results and insights:  
  - [Downsampling process](https://github.com/hicksjr2/ISA444-Final-Project/blob/main/ISA444_downsample_preprocess.ipynb)
  - [TimeCopilot Colab](https://colab.research.google.com/drive/1VxCQ1UMSyaKdJ48e8yJ2O1BFTTCi1wCR#scrollTo=lm0qjca4qiNh) *Use Google Colab to view code*
  - 

