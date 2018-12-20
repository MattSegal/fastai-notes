# Recurrent Neural Nets


### Financial Data

TODO

Can you clarify why "the target series should be log-normal returns" is important or provide a pointer for more information?

    Investment markets operate on relative gain, not absolute gain. E.g.: if you invest in a stock and it gains $5, this would be a great return for a $1 stock but a poor one for a $1000 stock, so the absolute gain doesn't mean anything on its own. A 5% return always means that you've gained 5% on your investment.

        Would it be the same (valid) with percentage returns?

            Less so. Log-normal returns are better because they have the property that a summation of log-normal returns over contiguous intervals is equal to the log-normal returns of the combined interval. In other words: Losing 5% and then gaining 5% doesn’t put you back at exactly 100%, and log-normal fixes that. In the extreme, two successive trades, where the first gains 110% and the second loses 100%, “average” out to a 5% return. However, you don’t want to make that pair of trades.

            https://financetrain.com/why-lognormal-distribution-is-used-to-describe-stock-prices/

            https://en.wikipedia.org/wiki/Stationary_process

https://blog.projectpiglet.com/2018/01/perils-of-backtesting/
https://otexts.org/fpp2/accuracy.html

Read Tsay's Financial Time Series Analysis for a better idea how to forecast financial data.
The biggest problem you will face is stationarity: the statistical properties of the data is not constant over time. For example, the mean and std dev is not constant over time. Using returns instead of raw prices helps to make better financial forecasts.

1. You are better off predicting stock prices by predicting future returns and then forecasting is the current price plus predicted future return.
2. You could use your neural model to predict absolute size of returns using realized volatility.


From my experience, LSTM or other recurrent neural network models only work "well" at forecasting bounded and periodic or oscillating time series. Might work for something like seasonal sale data, but would fail spectacularly with unbounded and chaotic time series like stock prices.

### RNNs

RNNs are useful for sequences of data. Why do we need RNNs?

- We want to keep track of long term dependencies between sequential observations. We need some kind of memory.
- We want to be able to ingest variable length sequences.
