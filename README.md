# Business Forecasting Using Statistical, ML, Deep Learning and Transformer Models

## Dataset
In this class project, we utilize the [Walmart Sales Forecasting dataset from Kaggle](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast) to compare a wide range of time-series forecasting models. 
Given that the full Walmart dataset contains hundreds of Store-Department series, we focused on the top 40 highest-volume time series. We selected these series because they have dense weekly histories, strong seasonal patterns, and stable sales behavior - making them ideal for comparing forecasting models.

## Forecasting Procedure
Our approach consisted of the following steps:  
  1. Loaded, cleaned & downsampled the Walmart dataset by selected the top 40 Store-Dept series ranked by total sales volume.
  2. Re-shaped the resulting dataset to match Nixtla's three-column panel format (unique_id, ds, y).
  3. Saved the cleaned dataset as [downsampled_df.csv], creating a consistent input for all forecasting models.
  4. We created forecasting models and focused on the naive and seasonal naive methods. These gave a realistic baseline for forecasting and allowed us to test the significance of seasonal patterns.
  5. We then used backtesting folds: each fold simulates a real forecasting scenario with training, predicting future weeks and comparing against actuals. For each fold, both models generated predictions for the 40 series.
  6. We then computed error metrics for every model, series and fold. We calculated the mean error, mean absolute error, root mean square error, and mean absolute percentage error. The results were combined to determine which model performed best for each store-dept pair.
  7. We identified model winners for each of the 40 series. We compared the naive and seasonal naive for every metric and found that the seasonal naive was the most accurate across ~ 80% of all the series depending on the metric.
  8. We then inspected the forecasts between actual and predicted weekly sales. We discovered that the seasonal naive method captures repetitive annual patterns and peaks during holidays while the naive missed these turning points.
  9. Finally, we combined our findings of the statistical performance measures, visuals, and model winner counts to solidify our insights about the behavior of retail demand and seasonal baselines. 

## Results and Observations
ABCD


## Our Python Notebooks
The following notebooks contain our code, results and insights:  
  - [Downsampling Process](https://github.com/hicksjr2/ISA444-Final-Project/blob/main/ISA444_downsample_preprocess.ipynb)
  - [Forecasting Models Colab](https://colab.research.google.com/drive/14YOiFIOjZzcY80prRuOus06nuGeIG7AE?usp=sharing) *Use Google Colab to view code*
  - 

