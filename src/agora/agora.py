#!/usr/bin/env python
import numpy as np
import pandas as pd
import seaborn as sns
import random
import re
import sys
from pandas_datareader import data as web
from datetime import datetime, date

import utils
from instrument import Instrument
from portfolio import Portfolio

tickers_data   = pd.DataFrame(pd.read_csv('data/tickers.csv'))
tickers        = tickers_data['Symbol'].to_frame()
stocks_tickers = list(tickers_data.dropna(subset = ['IPOyear'])['Symbol'])
##############################################
#               Ticker Symbols               #
##############################################

def get_tickers():
	'''
	function:
		This function displays all the available ticker symbols that start with `letter`

	arguments:
		[*] letter : the letter from which symbols start.  Available Options : 
					-> a - z, A - Z, all (everything)

	'''
	# 1 - check arguments
	if not utils.check_argv(3,"WARNING! Correct Usage: ./agora.py tickers <L>") :
		return

	letter = sys.argv[2]

	# 2 - check parameter validity
	try :
		letter.isalpha() or letter == 'all'
	except ValueError:
		raise ValueError("ERR#0012: There is no ticker starting with this letter.")

	# 3 - function

	if letter == 'all' : 
		tickers_to_display = tickers
	else:
		tickers_to_display = tickers.loc[ tickers['Symbol'].str.startswith(letter)]

	utils.display(tickers_to_display)
	if letter == 'all' : 
		print("Total no. of tickers  : {} / {}".format(len(tickers_to_display), len(tickers)))
	else:
		print("Total no. of tickers starting with '{}' : {} / {}".format(letter, len(tickers_to_display), len(tickers)))

	return

#############################################
#              Ticker : Data                #
#############################################
def get_ticker_historical_data(**kwargs):
	'''
	function:
		This function retrieves all the price data for a ticker in the following format :

            Date || Open | High | Low | Close | Adj Close |
            -----||------|------|-----|-------|-----------|
            xxxx || xxxx | xxxx | xxx | xxxxx | xxxxxxxxx |

    args:
    	[*] ticker    : The ticker for which historical data are retrieved
    	[*] from_date : Date formatted as `dd/mm/yyyy`, since when data is going to be retrieved.
    	[*] to_date   : Date formatted as `dd/mm/yyyy`, until when data is going to be retrieved.
	'''
	# 1 - check arguments
	if kwargs == {}:
		if not utils.check_argv(5,"WARNING! Correct Usage: ./agora.py ticker-data <ticker> <start> <end>") :
			return

		ticker = sys.argv[2]
		start  = sys.argv[3]
		end    = sys.argv[4]
		printing = True
	else:
		ticker = kwargs['ticker']
		start  = kwargs['start']
		end    = kwargs['end']
		printing = False

	# 2 - check parameter validity
	# TICKER
	try :
		ticker in tickers['Symbol']
	except ValueError:
		raise ValueError("ERR#0012: There is no ticker with that name. Check available tickers symbols with : `./agora.py tickers *`")

	# DATETIMES
	try:
		datetime.strptime(start, '%d/%m/%Y')
	except ValueError:
		raise ValueError("ERR#0001: incorrect start date format, it should be 'dd/mm/yyyy'.")
	try:
		datetime.strptime(end, '%d/%m/%Y')
	except ValueError:
		raise ValueError("ERR#0002: incorrect en dformat, it should be 'dd/mm/yyyy'.")

	start = datetime.strptime(start, '%d/%m/%Y')
	end = datetime.strptime(end, '%d/%m/%Y')


	if start >= end:
		raise ValueError("ERR#0003: `end` should be greater than `start`, both formatted as 'dd/mm/yyyy'.")

	date_range = {'start' : start, 'end' : end}

	# 3 - Retrieve instrument data
	instrument = Instrument(ticker, date_range)
	if printing : utils.display(instrument.data.tail(15))

	# 4 - print result
	messages = [ " Trading for T = {} days ".format(len(instrument.data) ) ]
	if printing : utils.pprint(messages)

	return instrument


###############################################################
#              Ticker : Descriptive Statistics                #
###############################################################
# [*] 1 TICKER
def get_ticker_statistics(**kwargs):
	'''
	function:
		This function 
		1. Uses `ticker_historical_data` to retrieve all the price data for a ticker in the following format :

            Date || Open | High | Low | Close | Adj Close |
            -----||------|------|-----|-------|-----------|
            xxxx || xxxx | xxxx | xxx | xxxxx | xxxxxxxxx |
        2. Calculate both RETURN & RISK descriptive statistics

    args:
    	[*] ticker    : The ticker for which historical data are retrieved
    	[*] start.    : Date formatted as `dd/mm/yyyy`, since when data is going to be retrieved.
    	[*] end       : Date formatted as `dd/mm/yyyy`, until when data is going to be retrieved.
	'''

	# 1 - check arguments
	if kwargs == {}:
		if not utils.check_argv(5,"WARNING! Correct Usage: ./agora.py ticker-statistics <ticker> <from> <to>") :
			return

		ticker = sys.argv[2]
		start  = sys.argv[3]
		end    = sys.argv[4]
		printing = True
	else:
		ticker = kwargs['ticker']
		start  = kwargs['start']
		end    = kwargs['end']
		printing = False

	# 2
	# 2.1 - Get the data
	instrument = get_ticker_historical_data(ticker = ticker, start = start, end = end )

	# 2.2 - Calculate [*] return 
	#                 [*] log return 
	#                 [*] expected daily return
	#                 [*] expected return
	instrument.calculate_statistics()
	return_statistics = instrument.return_statistics
	risk_statistics   = instrument.risk_statistics


	# 4 - print result
	# RETURN
	messages = []
	messages.append(" Expected Total Return  ({} days)  = {} %".format(len(instrument.data) , round(return_statistics['expected_total_return'] * 100, 3)))
	messages.append(" Expected Annual Return (252 days)  = {} % ".format(round(return_statistics['expected_annual_return'] * 100, 3)))
	messages.append(" APR = {} % ".format(round(return_statistics['APR'] * 100, 3)))
	messages.append(" APY = {} % ".format(round(return_statistics['APY'] * 100, 3)))
	if printing :  utils.pprint(messages)

	# RISK
	messages = []
	messages.append(" Total Standard Deviation  ({} days)  = {}  ".format(len(instrument.data), round(risk_statistics['total_std'], 3)))
	messages.append(" Annual Standard Deviation (252 days) = {} ".format(round(risk_statistics['annual_std'], 3)))
	messages.append(" Total Variance  ({} days)  = {}  ".format(len(instrument.data), round(risk_statistics['total_var'], 3)))
	messages.append(" Annual Variance (252 days) = {} ".format(round(risk_statistics['annual_var'], 3)))
	if printing : utils.pprint(messages)

	return instrument

# [*] N TICKERS
def get_tickers_statistics(**kwargs):
	'''
	function:
		This function 
		1. Uses `get_ticker_statistics` N times, 1 for each ticker instrument. For each instrument
			1.1 Uses `ticker_historical_data` to retrieve all the price data for a ticker in the following format :

            Date || Open | High | Low | Close | Adj Close |
            -----||------|------|-----|-------|-----------|
            xxxx || xxxx | xxxx | xxx | xxxxx | xxxxxxxxx |
        	1.2. Calculate both RETURN & RISK descriptive statistics

    args:
    	[*] N 			: Number of tickers
    	[*] ticker_list : The ticker list for which historical data are retrieved
    	[*] start       : Date formatted as `dd/mm/yyyy`, since when data is going to be retrieved.
    	[*] end         : Date formatted as `dd/mm/yyyy`, until when data is going to be retrieved.
	'''

	# 1 - check arguments
	if kwargs == {}:
		num_tickers = int(sys.argv[2])
	
		if not utils.check_argv(5 + num_tickers,"WARNING! Correct Usage: ./agora.py ticker-statistics <N> <ticker1> <ticker2> .. <tickerN> <from> <to>") :
			return

		ticker_list = []
		for i in range(3, num_tickers + 3):
			ticker_list.append(sys.argv[i])
		start       = sys.argv[num_tickers + 3]
		end         = sys.argv[num_tickers + 4]
		printing = True
	else:
		ticker_list = kwargs['ticker_list']
		start       = kwargs['start']
		end         = kwargs['end']
		printing= False

	# 2 - Retrieve Data & Calculate Descriptive statistics for each ticker:
	instrument_list             = []
	expected_annual_return_list = []
	annual_std_list             = []
	for ticker in ticker_list:
		instrument = get_ticker_statistics(ticker = ticker, start = start, end = end )
		instrument_list.append(instrument)
		expected_annual_return_list.append(instrument.return_statistics['expected_annual_return'] * 100)
		annual_std_list.append(instrument.risk_statistics['annual_std'])

	# 3 - Convert Descriptive statistics from list to dataframes
	descriptive_dict = {"Expected Annual Return" : expected_annual_return_list,
						"Annual Standard Deviation" : annual_std_list
						}
	descriptive_df = pd.DataFrame(descriptive_dict)
	descriptive_df.index = ticker_list

	# 4 - Display results
	utils.display(descriptive_df)

	return instrument_list, descriptive_df



###############################################################
#              Ticker : Risk Analysis Statistics              #
###############################################################
# [*] 1 TICKER
def get_ticker_risk_analysis(**kwargs):
	'''
	function:
		This function
		1. Uses `get_ticker_statistics` to calculate both RETURN & RISK descriptive statistics
		2. Applies 'Capital Asset Pricing Model' to calculate α (alpha),β (beta) , correlation ρ
		   For all this to succeed we define:
		   - risk-free rate : the 3month Tbill 
		   - Market Index   : S&P 500
	'''
	# 1 - check arguments
	if kwargs == {}:
		if not utils.check_argv(5,"WARNING! Correct Usage: ./agora.py ticker-risk-analysis <ticker> <from> <to>") :
			return

		ticker = sys.argv[2]
		start  = sys.argv[3]
		end    = sys.argv[4]
		printing = True
	else:
		ticker = kwargs['ticker']
		start  = kwargs['start']
		end    = kwargs['end']
		printing = False

	# 2
	# 2.1 - Get the data & descriptive statistics
	instrument = get_ticker_statistics(ticker = ticker, start = start, end = end )

	# 2.2 - Calculate [*] alpha α 
	#                 [*] beta β
	#                 [*] correlation of instrument & market ρ
	instrument.risk_analysis()
	risk_analysis_statistics = instrument.risk_analysis_statistics

	# 3 - print result
	# RISK
	messages = []
	messages.append(" Let's apply CAPM modelling for Risk Analysis [ Market : S&P500 , Instrument : {}] ".format(ticker))
	messages.append(" Correlation  [ρ]  = {}  ".format(round(risk_analysis_statistics['correlation'], 3)))
	messages.append(" Alpha        [α]  = {}  ".format(round(risk_analysis_statistics['alpha'], 3)))
	messages.append(" Beta         [β]  = {}  ".format(round(risk_analysis_statistics['beta'], 3)))
	messages.append(" Sharpe Ratio [SR]  = {} ".format(round(risk_analysis_statistics['sharpe_ratio'], 3)))
	messages.append(" R Squared    [R^2] = {} % ".format(round(risk_analysis_statistics['r_squared'] * 100, 3)))
	if printing : utils.pprint(messages)

	return instrument



# [*] N TICKERS
def get_tickers_risk_analysis(**kwargs):
	'''
	function:
		This function 
		1. Uses `get_ticker_risk_analysis` N times, 1 for each ticker instrument. For each instrument
			1.1. Uses `get_ticker_statistics` to calculate both RETURN & RISK descriptive statistics
			1.2. Applies 'Capital Asset Pricing Model' to calculate α (alpha),β (beta) , correlation ρ
			   For all this to succeed we define:
			   - risk-free rate : the 3month Tbill 
			   - Market Index   : S&P 500

    args:
    	[*] N 			: Number of tickers
    	[*] ticker_list : The ticker list for which historical data are retrieved
    	[*] start       : Date formatted as `dd/mm/yyyy`, since when data is going to be retrieved.
    	[*] end         : Date formatted as `dd/mm/yyyy`, until when data is going to be retrieved.
	'''

	# 1 - check arguments
	if kwargs == {}:
		num_tickers = int(sys.argv[2])
	
		if not utils.check_argv(5 + num_tickers,"WARNING! Correct Usage: ./agora.py ticker-risk-analysis <N> <ticker1> <ticker2> .. <tickerN> <from> <to>") :
			return

		ticker_list = []
		for i in range(3, num_tickers + 3):
			ticker_list.append(sys.argv[i])
		start       = sys.argv[num_tickers + 3]
		end         = sys.argv[num_tickers + 4]
		printing = True
	else:
		ticker_list = kwargs['ticker_list']
		start       = kwargs['start']
		end         = kwargs['end']
		printing    = False

	# 2 - Retrieve Data & Calculate Descriptive statistics for each ticker:
	get_tickers_statistics(ticker_list = ticker_list, start = start, end = end)
	instrument_list             = []
	alpha_list, beta_list, correlation_list = [], [], []
	sharpe_ratio_list, r_squared_list       = [], [] 

	for ticker in ticker_list:
		instrument = get_ticker_risk_analysis(ticker = ticker, start = start, end = end )
		alpha_list.append(instrument.risk_analysis_statistics['alpha'])
		beta_list.append(instrument.risk_analysis_statistics['beta'])
		correlation_list.append(instrument.risk_analysis_statistics['correlation'])
		sharpe_ratio_list.append(instrument.risk_analysis_statistics['sharpe_ratio'])
		r_squared_list.append(instrument.risk_analysis_statistics['r_squared'] * 100)


	# 3 - Convert Descriptive statistics from list to dataframes
	risk_analysis_dict = {"Alpha" : alpha_list,
						  "Beta"  : beta_list,
						  "Correlation with S&P500" : correlation_list,
						  "Sharpe Ratio" : sharpe_ratio_list,
						  "R^2" : r_squared_list
						}
	risk_analysis_df =  pd.DataFrame(risk_analysis_dict)
	risk_analysis_df.index = ticker_list


	# 4 - Display results
	utils.display(risk_analysis_df)

	return instrument_list, risk_analysis_df


#####################################################
#              Portfolio Optimization               #
#####################################################
'''
	" Modern Portfolio Theory " 

	[1] Modern Portfolio Theory (MPT) is an investment theory developed by Harry Markowitz and published 
        under the title "Portfolio Selection" in the Journal of Finance in 1952.

    [2] There are a few underlying concepts that can help anyone to understand MPT. If you are familiar with finance, 
        you might know what the acronym "TANSTAAFL" stands for. It is a famous acronym for "There Ain't No Such Thing As A Free Lunch".
        This concept is also closely related to 'risk-return trade-off'.

    [3] Higher risk is associated with greater probability of higher return and lower risk with a greater probability of smaller return. 
        MPT assumes that investors are risk-averse, meaning that given two portfolios that offer the same expected return,  investors 
        will prefer the less risky one. Thus, an investor will take on increased risk only if compensated by higher expected returns.

    [4] Another factor comes in to play in MPT is "diversification". Modern portfolio theory says that it is not enough to look at 
        the expected risk and return of one particular stock. By investing in more than one stock, an investor can reap the benefits 
        of diversification – chief among them, a reduction in the riskiness of the portfolio.

    [5] What you need to understand is "risk of a portfolio is not equal to average/weighted-average of individual stocks in the portfolio". 
    	[*] PORTFOLIO RETURN : Yes it is the average/weighted average of individual stock's returns, but that's not the case for risk. 
			                                      
			                                      ###################################################################
    											  #    E[R_p] = w_1 * E[R_1] + w_2 * E[R_2] + ... + w_n * E[R_n]    #
    											  ###################################################################

    	[*] PORTFOLIO RISK   : The risk is about how volatile the asset is, if you have more than one stock in your portfolio, then you 
    						   have to take count of how these stocks movement correlates with each other. The beauty of diversification 
    						   is that you can even get lower risk than  a stock with the lowest risk in your portfolio, by optimising the allocation.
											 	   
											 	  ###############################################################################
											 	  #    							+-------------------------+	  +---+				#
											 	  #    							|cov_11,cov_12 ... ,cov_1n|	  |w_1|				#
											 	  #    							|cov_21,cov_22 ... ,cov_2n|	  |w_2|				#
    											  #    σ_p = [w_1, ... , w_n] * |              ...        | * |...|				#
    											  #    							|              ...        |	  |...|				#
    											  #    							|cov_n1,cov_n2 ... ,cov_nn|	  |w_n|				#
    											  #	    					 	+-------------------------+	  +---+				#
    											  ###############################################################################
'''
def portfolio_construction(**kwargs):
	'''
	function:
		This function :
		1. Uses `get_tickers_statistics` N times, 1 for each ticker instrument to calculate the descriptive metrics
		2. Capital Allocation : Randomly. In other words . One decision we have to make is how we should allocate our 
		                        budget to each of instrument in our portfolio. If our total budget is 1, then we can 
		                        decide the weights for each stock, so that the sum of weights will be 1. And the value 
		                        for weights will be the portion of budget we allocate to a specific stock. For example, 
		                        if weight is 0.5 for Amazon, it means that we allocate 50% of our budget to Amazon.

	args:
		[*] N 			: Number of tickers
    	[*] ticker_list : The ticker list of which portfolio will be constructed
    	[*] start       : Date formatted as `dd/mm/yyyy`, since when data is going to be retrieved.
    	[*] end         : Date formatted as `dd/mm/yyyy`, until when data is going to be retrieved.
	'''
	# 1 - check arguments
	if kwargs == {}:
		num_tickers = int(sys.argv[2])
	
		if not utils.check_argv(5 + num_tickers,"WARNING! Correct Usage: ./agora.py portfolio-construction <N> <ticker1> <ticker2> .. <tickerN> <from> <to>") :
			return

		ticker_list = []
		for i in range(3, num_tickers + 3):
			ticker_list.append(sys.argv[i])
		start       = sys.argv[num_tickers + 3]
		end         = sys.argv[num_tickers + 4]
		printing = True
	else:
		ticker_list = kwargs['ticker_list']
		start       = kwargs['start']
		end         = kwargs['end']
		printing    = False

	# [2.0] - Get the risk-free & returns merged
	# [2.1] - Get the descriptive statistics for the N tickers/instruments 
	# [2.2] - Create a portfolio object.
	# [2.3] - Initialize weights
	# [2.4] - Calculate portfolio descriptice statistics
	instrument_list, descriptive_df = get_tickers_statistics(ticker_list = ticker_list, start = start, end = end)
	risk_free                       = utils.risk_free_return(date_range = instrument_list[0].date_range)
	returns_merged                  = utils.merge_instrument_returns(instrument_list = instrument_list, ticker_list = ticker_list)
	portfolio = Portfolio(instrument_list = instrument_list, ticker_list = ticker_list, returns_merged = returns_merged, risk_free = risk_free )

	#-------------------------------------------------------------------------------------------------------------------------#
	# [3] - show the capital allocation (weights) & the statistics
	message = "              * Random Portfolio *            "
	portfolio.initialize_weights()
	portfolio.calculate_statistics()
	portfolio.track_progress(printing, message, True)

	message = "          * Unoptimized Risky Portfolio (stocks = 0.45, bonds = 0.35, commodities = 0.1) *           "
	portfolio.weights  = np.array([0.45/7] * 7 + [0.35/2] * 2 + [0.1/2] * 2).flatten()
	portfolio.calculate_statistics()
	portfolio.track_progress(printing, message, True)

	message = "           * Unoptimized Total Portfolio (stocks = 0.45, bonds = 0.35, commodities = 0.1, risk-free = 0.1) *           "
	portfolio.track_progress(printing, message, False)

	#-------------------------------------------------------------------------------------------------------------------------#
	# [3] - Plot the data points for these 3 portfolios (random, risky, total portfolio)
	title = "initial_portfolios"
	portfolio_arr = ["Random", "Unoptimized Risky", "Unoptimized Total"]
	portfolio.plot_initial_portfolios(title, portfolio_arr, descriptive_df)


	return portfolio



def portfolio_optimization(**kwargs):
	'''
	function:
		This function :
		1. Uses `get_tickers_statistics` N times, 1 for each ticker instrument to calculate the descriptive metrics
		2. Calls `portfolio_construction` for `num_port` times to construct `num_port` random portfolios
			2.1 Initialize random weights for the corresponding porftolio. 
		3. Then by locating the one with the highest Sharpe ratio portfolio, it displays 
				[*] Maximum Sharpe ratio portfolio as red star sign. 
				[*] Minimum volatility portfolio as green start sign
			All the randomly generated portfolios will be also plotted  with colour map applied to them based on the Sharpe ratio. 
			The bluer, the higher Sharpe ratio.
		4. For these two optimal portfolios, it will also show how it allocates the budget within the portfolio.
		

	args:
		[*] P 			: Number of portfolios
		[*] N 			: Number of tickers
    	[*] ticker_list : The ticker list of which portfolio will be constructed
    	[*] start       : Date formatted as `dd/mm/yyyy`, since when data is going to be retrieved.
    	[*] end         : Date formatted as `dd/mm/yyyy`, until when data is going to be retrieved.
	'''
	# 1 - check arguments
	if kwargs == {}:
		num_portfolios = int(sys.argv[2])
		num_tickers = int(sys.argv[3])
	
		if not utils.check_argv(6 + num_tickers,"WARNING! Correct Usage: ./agora.py portfolio-construction <N> <ticker1> <ticker2> .. <tickerN> <from> <to>") :
			return

		ticker_list = []
		for i in range(4, num_tickers + 4):
			ticker_list.append(sys.argv[i])
		start       = sys.argv[num_tickers + 4]
		end         = sys.argv[num_tickers + 5]
		printing = True
	else:
		num_portfolios  = kwargs['num_portfolios']
		ticker_list 	= kwargs['ticker_list']
		start       	= kwargs['start']
		end         	= kwargs['end']
		printing    	= False

	# 2 - Get the instrument list along with their calculated descriptive statistics
	instrument_list, descriptive_df = get_tickers_statistics(ticker_list = ticker_list, start = start, end = end)
	stocks_idx                      = [ idx for idx in range(len(ticker_list)) if ticker_list[idx] in stocks_tickers]
	risk_free                       = utils.risk_free_return(date_range = instrument_list[0].date_range)
	returns_merged                  = utils.merge_instrument_returns(instrument_list = instrument_list, ticker_list = ticker_list)

	# 3 - Portfolio simulation
	all_weights, ret_arr, std_arr, sharpe_arr = [], [], [], []
	for i in range(num_portfolios):
		if i % 100 == 0: print("{} out of {}\n".format(i, num_portfolios), end = '')
		portfolio = Portfolio(instrument_list = instrument_list, returns_merged = returns_merged , 
							  ticker_list = ticker_list, risk_free = risk_free)

		# weights
		portfolio.initialize_weights()
		w_stocks = sum([ w[i] for i in portfolio.weights if i in stocks_idx ])

		# return, std, sharpe ratio
		portfolio.calculate_statistics()
		portfolio_statistics = portfolio.statistics
		R_P   = portfolio_statistics['portfolio_annual_return']
		STP_P = portfolio_statistics['portfolio_annual_std']
		SR_P  = portfolio_statistics['portfolio_annual_sr']

		# append results
		all_weights.append(portfolio.weights)
		ret_arr.append(R_P)
		std_arr.append(STP_P)
		sharpe_arr.append(SR_P)

	# 4 - Calculate 2 most efficient portfolios.
	# [1] Max Sharpe Ratio Portfolio
	opt_idx = np.argmax(sharpe_arr)
	opt_sr, opt_ret, opt_std = sharpe_arr[opt_idx], ret_arr[opt_idx], std_arr[opt_idx]
	opt_weights = all_weights[opt_idx]
	messages = []
	messages.append("          * Max Sharpe Ratio optimized Portfolio *          ")
	messages.append(" Portfolio Annual Return (252 days)  = {} % ".format(round(opt_ret * 100, 3)))
	messages.append(" Portfolio Annual Standard Deviation  (252 days)  = {}  ".format( round(opt_std, 3)))
	messages.append(" Portfolio Annual Sharpe Ratio  (252 days)  = {}  ".format( round(opt_sr, 3)))
	if printing : utils.pprint(messages)

	# [2] Min Standard Deviation Ratio portfolio
	min_idx = np.argmin(std_arr)
	min_sr, min_ret, min_std = sharpe_arr[min_idx], ret_arr[min_idx], std_arr[min_idx]
	min_weights = all_weights[min_idx]
	messages = []
	messages.append("      * Min Standard Deviation optimized Portfolio *      ")
	messages.append(" Portfolio Annual Return (252 days)  = {} % ".format(round(min_ret * 100, 3)))
	messages.append(" Portfolio Annual Standard Deviation  (252 days)  = {}  ".format( round(min_std, 3)))
	messages.append(" Portfolio Annual Sharpe Ratio  (252 days)  = {}  ".format( round(min_sr, 3)))
	if printing : utils.pprint(messages)

	# [3] weight allocation for both efficient portfolios
	weights_dict = {"Max SR Allocation Weights" : opt_weights * 100, 'Min σ Allocation Weights' : min_weights * 100}
	weights_df =  pd.DataFrame(weights_dict)
	weights_df.index = ticker_list
	if printing : utils.display(weights_df)

	# 5 - Plot the portfolios along with the 2 efficient portfolios.
	title = "{}_portfolio_simulation".format(num_portfolios)
	portfolio.plot_portfolio_simulation(title, instrument_list[0].date_range, std_arr,ret_arr, sharpe_arr, descriptive_df, returns_merged)


	return



def help():
	print("=========================================================================")
	print("=                            Commands Available                         =")
	print("=========================================================================")
	print("       1. tickers <letter>			            	 					  Get all tickers list that starts with <letter>")
	print("       2. ticker-data <ticker> <from> <to>			 					  Get all historical data for <ticker> from date <from> to date <to>")
	print("       3.1 ticker-statistics <ticker> <from> <to>      					  Get all historical data & descriptive statistics for <ticker> from date <from> to date <to>")
	print("       3.2 tickers-statistics <N> <ticker1> ... <tickersN> <from> <to>     Get all historical data & descriptive statistics for <ticker1>, ... <tickerN> from date <from> to date <to>")
	print("       4.1 ticker-risk-analysis <ticker> <from> <to>   					  Get all historical data & descriptive  & risk-analysis statistics for <ticker> from date <from> to date <to>")
	print("       4.2 tickers-risk-analysis <N> <ticker1> ... <tickersN> <from> <to>  Get all historical data & descriptive  & risk-analysis statistics for <ticker1>, ... <tickerN> from date <from> to date <to>")
	print("       5.1 portfolio-construction <N> <ticker1> ... <tickersN> <from> <to> Construct a portfolio with the <N> instruments <ticker1>, ... <tickerN> from date <from> to date <to>")
	print("       5.2 portfolio-optimization <N> <ticker1> ... <tickersN> <from> <to> Optimize a portfolio with the <N> instruments <ticker1>, ... <tickerN> from date <from> to date <to>")
	
	return




if __name__ == '__main__':
	print("=========================================================================")
	print("=                                                                       =")
	print("=                          Welcome to Agora [!]                         =")
	print("=                                                                       =")
	print("=========================================================================")

	if len(sys.argv) == 1:
		print('\nWARNING!! No system command given. Check available commands with help() \n')
		exit()

	cmd = sys.argv[1]
	
	if   cmd == 'tickers'                : get_tickers()
	elif cmd == 'ticker-data'            : get_ticker_historical_data()
	elif cmd == 'ticker-statistics'      : get_ticker_statistics()
	elif cmd == 'tickers-statistics'     : get_tickers_statistics()
	elif cmd == 'ticker-risk-analysis'   : get_ticker_risk_analysis()
	elif cmd == 'tickers-risk-analysis'  : get_tickers_risk_analysis()
	elif cmd == 'portfolio-construction' : portfolio_construction()
	elif cmd == 'portfolio-optimization' : portfolio_optimization()
	elif cmd == 'help'                   : help()
	else :
		print('Command not found')
