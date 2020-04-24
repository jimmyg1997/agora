
import numpy as np
import pandas as pd
import seaborn as sns
import random
import re
import sys
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import solvers
from pandas_datareader import data as web
from datetime import datetime, date
import scipy.optimize as sco
import utils
class Portfolio():

	def __init__(self, **kwargs):
		self.instrument_list    = kwargs['instrument_list']
		self.returns_merged     = kwargs['returns_merged']
		self.ticker_list        = kwargs['ticker_list']
		self.num_instruments    = len(self.instrument_list)
		self.risk_free          = kwargs['risk_free']

		self.weights            = []
		self.statistics         = {}
		self.covariance_matrix  = []

		return

	def initialize_weights(self):
		'''
		function:
			For this portfolio we generate random initial weights that are normalized so that : 
								######################################
								##         sum(weights) = 1         ##
								######################################

		'''
		weights = np.random.random(self.num_instruments)

		self.weights = weights / np.sum(weights)
		return

	def calculate_statistics(self):

		statistics = {}

		#################################################################################
    	# 																				#
    	# [1] E[R_p] = w_1 * E[R_1] + w_2 * E[R_2] + ... + w_n * E[R_n]                 #
    	#																				#
 	  	#    							+-------------------------+	  +---+				#
 	  	#    							|cov_11,cov_12 ... ,cov_1n|	  |w_1|				#
 	  	#    							|cov_21,cov_22 ... ,cov_2n|	  |w_2|				#
	  	#  [2] σ_p = [w_1, ... , w_n] * |              ...        | * |...|				#
	  	#    							|              ...        |	  |...|				#
	  	#    							|cov_n1,cov_n2 ... ,cov_nn|	  |w_n|				#
	  	#	    					 	+-------------------------+	  +---+				#
	  	#				E[R_p] - R_rf 												    #
	  	#  [3] SR_p = ----------------													#
	  	#                   σ_p															#
	  	#################################################################################
	  	# PORTFOLIO RETURN
		R_I_list   = [instrument.return_statistics['expected_annual_return'] for instrument in self.instrument_list]
		statistics["portfolio_annual_return"] = np.sum(R_I_list * self.weights)

		# PORFTOLIO STANDARD DEVIATION
		STD_I_list = [instrument.risk_statistics['annual_std'] for instrument in self.instrument_list]
		covariance_matrix = self.calculate_covariance_matrix()
		statistics["portfolio_annual_std"] = np.sqrt(np.dot(self.weights.T, np.dot(covariance_matrix, self.weights))) * np.sqrt(252)


		# PORFTOLIO SHARPE RATIO
		statistics["portfolio_annual_sr"] = (statistics["portfolio_annual_return"] - self.risk_free) / statistics["portfolio_annual_std"]

		self.statistics = statistics

		return

	def calculate_covariance_matrix(self):
		'''
		function:
			1. This functions takes the list of returns for all N instruments : 
				              [ returns_df1 , ..., returns_dfn ]

			2. Merges all dataframes (inner join) based on the index = Dates & creating a new dataframe 
			   which stores the returns for all tickers at the COMMMON TRADING DAYS.
							 merge([ returns_df1 , ..., returns_dfn ])
			Steps 1, 2 are done in utils.

			3. Finally calculates the covariance matrix which is N x N. This shows the covariance between
			   a pair of instruments. For example 
			   covariance_matrix[i,j] = covariance between instrument i & instrument j

		'''
		covariance_matrix = self.returns_merged.cov()
		return covariance_matrix

	def track_progress(self, printing, message_optimization, risky):

		portfolio_statistics = self.statistics

		weights = self.weights 
		weights_df =  pd.DataFrame(weights, columns = ['Allocation Weights'])
		weights_df.index = self.ticker_list

		annual_return = portfolio_statistics['portfolio_annual_return']
		annual_std    = portfolio_statistics['portfolio_annual_std']
		annual_sr     = portfolio_statistics['portfolio_annual_sr']

		if not risky : 
			annual_return = annual_return * 0.9 + self.risk_free * 0.1
			annual_std    = 0.9 ** 2 * annual_std
			annual_sr     = (annual_return - self.risk_free) / annual_std

		utils.portfolios['ret'].append(annual_return)
		utils.portfolios['std'].append(annual_std)
		utils.portfolios['sr'].append(annual_sr)

		messages = []
		messages.append(message_optimization)
		messages.append(" Portfolio Annual Return (252 days)  = {} % ".format(round(annual_return * 100, 3)))
		messages.append(" Portfolio Annual Standard Deviation  (252 days)  = {}  ".format( round(annual_std, 3)))
		messages.append(" Portfolio Annual Sharpe Ratio  (252 days)  = {}  ".format( round(annual_sr, 3)))

		if printing : utils.display(weights_df.T)
		if printing : utils.pprint(messages)

		return



	def portfolio_annualised_performance(self, weights, mean_returns, cov_matrix):

		portfolio_return = np.sum(mean_returns * weights) * 252
		portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
		sharpe_ratio = (portfolio_return - self.risk_free) / portfolio_std
		return -sharpe_ratio


	def efficient_sharpe_ratio(self, mean_returns, cov_matrix, target):
		'''
			[*] Firstly, as we will be using the ‘SLSQP’ method in our “minimize” function (which stands for Sequential 
				Least Squares Programming), the constraints argument must be in the format of a list of dictionaries, 
				containing the fields “type” and “fun”, with the optional fields “jac” and “args”. We only need the fields 
				“type”, “fun” and “args” so lets run through them.

			[*]	The “type” can be either “eq” or “ineq” referring to “equality” or “inequality” respectively. The “fun” refers 
				to the function defining the constraint, in our case the constraint that the sum of the stock weights must be 1. 
				The way this needs to be entered is sort of a bit “back to front”. The “eq” means we are looking for our function 
				to equate to zero (this is what the equality is in reference to – equality to zero in effect). So the most simple 
				way to achieve this is to create a lambda function that returns the sum of the portfolio weights, minus 1. 
				The constraint that this needs to sum to zero (that the function needs to equate to zero) by definition means 
				that the weights must sum to 1. It’s admittedly a bit strange looking for some people at first, but there you go…

			[*]	The “bounds” just specify that each individual stock weight must be between 0 and 1, with the “args” being the arguments 
				that we want to pass to the function we are trying to minimise (calc_neg_sharpe) – that is all the arguments EXCEPT the 
				weights vector which of course is the variable we are changing to optimise the output.
		'''
		num_assets = len(mean_returns)
		args = (mean_returns, cov_matrix)

		constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

		bounds = tuple((0,1) for asset in range(num_assets))
		result = sco.minimize(self.portfolio_annualised_performance, num_assets * [1./num_assets,], args = args, method = 'SLSQP', 
							  bounds = bounds, constraints = constraints)
		return result

	def capital_allocation_line(self, ret_arr, opt_sr):
		'''
		function:
			Calculate x, y of the Capital Allocation Line (CAL)
		'''

		##################################################################
		#						     E[R_optimal] - R_rf 				 #
		#			E[R_p] = R_rf + ---------------------- σ_p 			 #
		#									σ_optimal				     #
		#							|                     |              #
		#                           |_____________________|              #
		#                                  SR_optimal                    #
		##################################################################
		A = 5
		cal_x , cal_y, utility = [], [], []

		for E_p in np.linspace(self.risk_free, max(ret_arr) + 0.1, 20):
			S_p = (E_p - self.risk_free) / opt_sr
			U_p = E_p - 0.5 * A * (S_p)**2

			cal_x.append(S_p)
			cal_y.append(E_p)
			utility.append(U_p)

		return cal_x, cal_y, utility


	def efficient_frontier(self, mean_returns, cov_matrix, returns_range):
		'''
		 The first function "efficient_sharpe_ratio" is calculating the most efficient portfolio for a given target return, and the second 
		 function "efficient_frontier" will take a range of target returns and compute efficient portfolio for each return level.
		 '''
		efficients = []
		for ret in returns_range:
			efficients.append(self.efficient_sharpe_ratio(mean_returns, cov_matrix, ret))
		return efficients

	def efficient_frontier2(self, returns):

		cov = np.matrix(returns.cov())
		n = returns.shape[1]
		avg_ret = np.matrix(returns.mean()).T
		r_min = 0.01
		mus = []
		for i in range(120):
			r_min += 0.0001
			mus.append(r_min)
		P = opt.matrix(cov)
		q = opt.matrix(np.zeros((n, 1)))
		G = opt.matrix(np.concatenate((- np.transpose(np.array(avg_ret)),  - np.identity(n)), 0))
		A = opt.matrix(1.0, (1,n))
		b = opt.matrix(1.0)
		opt.solvers.options['show_progress'] = False
		portfolio_weights = [solvers.qp(P, q, G, opt.matrix(np.concatenate((-np.ones((1,1))*yy, np.zeros((n,1))), 1)), A, b)['x'] for yy in mus]
		portfolio_returns = [(np.matrix(x).T * avg_ret)[0,0] for x in portfolio_weights]
		portfolio_stdvs = [np.sqrt(np.matrix(x).T * cov.T.dot(np.matrix(x)))[0,0] for x in portfolio_weights]
		return portfolio_stdvs, portfolio_returns

	def plot_portfolio_simulation(self, title, date_range, std_arr,ret_arr, sharpe_arr, descriptive_df, returns_merged):
		# [1] Define the data points for :
		# # 1.0 - Efficient Frontier
		# mean_returns = returns_merged.mean() 
		# cov_matrix  = returns_merged.cov()
		# efficient_x, efficient_y = self.efficient_frontier2(returns_merged)
		
		# target = np.linspace(0.2, 0.46, 100)
		# efficient_portfolios = self.efficient_frontier(mean_returns, cov_matrix, target)

		# 1.1 - individual instruments
		y_I = R_I = descriptive_df["Expected Annual Return"] / 100
		x_I = STD_I = descriptive_df["Annual Standard Deviation"]

		# 1.2 - Efficient porftolios
		opt_idx = np.argmax(sharpe_arr)
		opt_sr, opt_ret, opt_std = sharpe_arr[opt_idx], ret_arr[opt_idx], std_arr[opt_idx]
		min_idx = np.argmin(std_arr)
		min_sr, min_ret, min_std = sharpe_arr[min_idx], ret_arr[min_idx], std_arr[min_idx]

		# 1.3 -  Capital Allocation Line (CAL)
		cal_x, cal_y, utility = self.capital_allocation_line(ret_arr ,opt_sr)

		# 1.4 - Market portfolio
		market = utils.market_info(date_range)
		R_M   = market["expected_annual_return_M"]
		STD_M = market["annual_std_M"]

		#----------------------------------------------------#

	 	# [2] Plot different data points
		plt.figure(figsize = (12,8))
	 	# 2.0 - efficient frontier
		# plt.plot(efficient_x, efficient_y, linestyle = '-.', color = 'black', label = 'efficient frontier')
		# plt.plot([p['fun'] for p in efficient_portfolios], target, linestyle = '-.', color = 'black', label = 'efficient frontier')

	 	# 2.1 - plot all random porftolios
		plt.scatter(std_arr, ret_arr, c = sharpe_arr, cmap = 'viridis', marker = 'o', s = 10, alpha = 0.5)
		plt.colorbar(label = 'Sharpe Ratio')

		# 2.2 - plot the 2 efficient porftolios : [*] Max Sharpe Ratio 
		#                                         [*] Min Standard Deviation
		plt.scatter(opt_std, opt_ret,marker = (5,1,0),color = 'r',s = 500, label = 'Max Sharpe ratio')
		plt.scatter(min_std, min_ret, marker = (5,1,0), color = 'g',s = 500,  label = 'Min Volatility ratio')

		# 2.3 - plot Market portfolio
		# plt.scatter(STD_M, R_M, s = 200 ,  alpha = 0.4, edgecolors = "grey", linewidth = 2)
		# plt.annotate('S&P500', (STD_M - 0.015, R_M - 0.02 ), size = 7.5)

		# 2.4 - plot Market CAL
		#plt.plot(cal_x, cal_y, linestyle = '-', color = 'red', label = 'CAL')

		# 2.5 - plot rest individual instruments
		# plt.scatter(x_I, y_I, s = 100 ,  alpha = 0.4, edgecolors = "grey", linewidth = 2)
		# for i, txt in enumerate(list(descriptive_df.index)):
		# 	plt.annotate(txt, (x_I[i] - 0.01 , y_I[i] - 0.025 ), size = 7.5)

		plt.title('Simulated portfolios')
		plt.xlabel('Annualized standard deviation')
		plt.ylabel('Annualized returns')
		plt.legend(labelspacing = 1.2)

		# [3] Save the graph
		plt.savefig(title, bbox_inches = 'tight')

		print(utility)

		return

	def plot_initial_portfolios(self, title, portfolio_arr, descriptive_df):
		# [1] Get Data
		# 1.1 - Initial 3 portfolios
		ret_arr, std_arr, sr_arr = utils.portfolios.values()

		# 1.2 - individual instruments
		y_I = R_I = descriptive_df["Expected Annual Return"] / 100
		x_I = STD_I = descriptive_df["Annual Standard Deviation"]

	 	# [2] Plot all porftolios
		axes = plt.gca()
		plt.figure(figsize = (12,8))

		# 2.1 - Plot initial 3 portfolios
		plt.scatter(std_arr, ret_arr, c = 'y', marker = 'o', edgecolors = "grey" , s = 100, alpha = 0.5, linewidth = 2)
		for i, txt in enumerate(portfolio_arr):
			plt.annotate(txt, (std_arr[i] + 0.01 , ret_arr[i] - 0.003), size = 7.5)
			plt.annotate("SR = {}".format(round(sr_arr[i],2)), (std_arr[i] - 0.007 , ret_arr[i] + 0.008 ), size = 7.5)

		# 2.2 - Plot individual instruments
		plt.scatter(x_I, y_I, s = 100 ,  alpha = 0.4, edgecolors = "grey", linewidth = 2)
		for i, txt in enumerate(list(descriptive_df.index)):
			plt.annotate(txt, (x_I[i] - 0.007, y_I[i] - 0.0155 ), size = 7.5)
			plt.annotate("SR = {}".format(round((y_I[i] - self.risk_free)/x_I[i],2)), (x_I[i] - 0.007 , y_I[i] + 0.009 ), size = 7.5)

		plt.title('Initial Portfolios')
		plt.xlabel('Annualized Standard Deviation')
		plt.ylabel('Annualized Returns')
		axes.set_facecolor((0.95, 0.95, 0.99))
		plt.grid(c = (0.75, 0.75, 0.99))

		# [3] Save the graph
		plt.savefig(title, bbox_inches = 'tight')

		return




