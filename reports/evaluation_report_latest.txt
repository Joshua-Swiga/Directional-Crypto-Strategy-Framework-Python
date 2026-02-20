Trading Strategy - Model Evaluation Report
Generated: 2026-02-19 14:33:47

SUMMARY
Model | Target | RMSE | MAE | R2 | MAPE%
----- | ------ | ---- | --- | -- | -----
ensemble | close | 8392.142131 | 6759.952386 | 0.0773 | 9.73
ensemble | returns | 0.004674 | 0.003155 | -0.1976 | 10099.81
linear_regression | close | 308.194786 | 196.171071 | 0.9988 | 0.27
linear_regression | returns | 0.004280 | 0.002647 | -0.0042 | 169.61
neural_network | close | 15376.633800 | 11481.280226 | -2.0977 | 16.72
neural_network | returns | 0.008145 | 0.005881 | -2.6361 | 3949.16
random_forest | close | 10487.731133 | 8257.522400 | -0.4410 | 11.94
random_forest | returns | 0.006753 | 0.005113 | -1.4991 | 1353.19
svr | close | 19420.660585 | 17039.812657 | -3.9413 | 24.07
svr | returns | 0.010136 | 0.009399 | -4.6305 | 49874.88

ERRORS
- xgboost: [Errno 2] No such file or directory: 'c:\\Users\\JoshuaSwiga\\Desktop\\strat\\Models\\xgboost\\xgboost_close.pkl'
- arima: [Errno 2] No such file or directory: 'c:\\Users\\JoshuaSwiga\\Desktop\\strat\\Models\\arima\\arima_close.pkl'
