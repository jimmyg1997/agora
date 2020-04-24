# 📈 Agora - Financial Markowitz Portfolio Optimization

This repository contains the Final Project for Deree course "Investments & Portfolio Management".

Authors:

* *Dimitrios Georgiou*
* *Angelos Pappas*

## What is *agora* ?

*Agora* is system that implements Markowitz Portfolio Optimization method (classical mean-variance technique). 

It is extensible, and can be useful for a casual investor giving initial insights through as a serious *Financial Instrument Analysis* or *Portfolio Management*.

Head over to the [report](https://raw.githubusercontent.com/jimmyg1997/NTUA-Multi-Criteria-Decision-Analysis/master/report.pdf) to get an in-depth look at the project.



<img src="https://raw.githubusercontent.com/jimmyg1997/agora/master/photos/system.png" alt="https://raw.githubusercontent.com/jimmyg1997/agora/master/photos/system.png">

## Getting Started

Commands are given via the terminal.
```
Usage :
$ ./agora.py COMMAND

Available Commands:
1   - `tickers` [letter]
2   - `ticker-data` [ticker] [start_date] [end_date]
3.1 - `ticker-statistics`[ticker1] ... [tickerN] [start_date] [end_date]
3.2 - `tickers-statistics`[ticker] [start_date] [end_date]
4.1 - `ticker-risk-analysis`[ticker] [start_date] [end_date]
4.2 - `tickers-risk-analysis`[ticker1] ... [tickerN] [start_date] [end_date]
5.1 - `portfolio-construction` [N] [ticker1] ... [tickerN] [start_date] [end_date]
5.2 - `portfolio-optimization` [N] [ticker1] ... [tickerN] [start_date] [end_date]
6   - `help`

Explanation:
1   - Get all tickers list that starts with <letter>
2.  - Get all historical data for <ticker> from <start_date> to <end_date>
3.1 - Get all historical data & descriptive statistics for <ticker> from <start_date> to <end_date>
3.2 - Get all historical data & descriptive statistics for <ticker1>, ... <tickerN> from <start_date> to <end_date>
4.1 - Get all historical data & descriptive  & risk-analysis statistics for <ticker> from <start_date> to <end_date>
4.2 - Get all historical data & descriptive  & risk-analysis statistics for <ticker1>, ... <tickerN> from <start_date> to <end_date>
5.1 - Construct a portfolio with <N> instruments <ticker1>, ... <tickerN> from <start_date> to <end_date>
5.2 - Optimize a portfolio with <N> instruments <ticker1>, ... <tickerN> from <start_date> to <end_date> through the Simulation of <P> portfolios
6   - Prints all available commands to the user


```


## Future Work

* **Optimization Techniques**
  * *Black Litterman Allocation as optimization technique*
  * *Hierarchical Risk Parity*
* **Machine Learning Techniques**
* **Macroeconomic & Microeconomic Analysis**
  * *Principal Component Analysis (PCA)* : Analyze and keep *N* most valuable features that caputre information about a financial instrument.
  * *Experimental Features* : exponentially-weighted covariance matrices. Etc.
