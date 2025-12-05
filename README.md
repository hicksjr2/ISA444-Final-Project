# Business Forecasting Using Statistical, ML, Deep Learning and Transformer Models

## Dataset
In this class project, we utilize the [Walmart Sales Forecasting dataset from Kaggle](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast) to compare a wide range of time-series forecasting models. 
Given that the full Walmart dataset contains hundreds of Store-Department series, we focused on the top 40 highest-volume time series. We selected these series because they have dense weekly histories, strong seasonal patterns, and stable sales behavior - making them ideal for comparing forecasting models.

## Forecasting Procedure
Our approach consisted of the following steps:  
### Downsampling Process
  1. Loaded, cleaned & downsampled the Walmart dataset by selected the top 40 Store-Dept series ranked by total sales volume.
  2. Re-shaped the resulting dataset to match Nixtla's three-column panel format (unique_id, ds, y).
  3. Saved the cleaned dataset as [downsampled_df.csv], creating a consistent input for all forecasting models.
### TimeCoPilot Colab
  4. .
  5. .
  6. .
  7. We created forecasting models and focused on the naive and seasonal naive methods. These gave a realistic baseline for forecasting and allowed us to test the significance of seasonal patterns. 
  8. We then used backtesting folds: each fold simulates a real forecasting scenario with training, predicting future weeks and comparing against actuals. For each fold, both models generated predictions for the 40 series.
  9. We then computed error metrics for every model, series and fold. We calculated the mean error, mean absolute error, root mean square error, and mean absolute percentage error. The results were combined to determine which model performed best for each store-dept pair.
  10. We identified model winners for each of the 40 series. We compared the naive and seasonal naive for every metric and found that the seasonal naive was the most accurate across ~ 80% of all the series depending on the metric.
  11. We then inspected the forecasts between actual and predicted weekly sales. We discovered that the seasonal naive method captures repetitive annual patterns and peaks during holidays while the naive missed these turning points.
  12. Finally, we combined our findings of the statistical performance measures, visuals, and model winner counts to solidify our insights about the behavior of retail demand and seasonal baselines. 

## Results and Observations
ABCD


## Our Python Notebooks
The following notebooks contain our code, results and insights:  
  - [Downsampling process](https://github.com/hicksjr2/ISA444-Final-Project/blob/main/ISA444_downsample_preprocess.ipynb)
  - [TimeCopilot Colab](https://colab.research.google.com/drive/1VxCQ1UMSyaKdJ48e8yJ2O1BFTTCi1wCR#scrollTo=lm0qjca4qiNh) *Use Google Colab to view code*
  - 

