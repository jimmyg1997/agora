
import numpy as np
import pandas as pd
import seaborn as sns
import random
import re
import sys
from pandas_datareader import data as web
from datetime import datetime, date

import utils
class Instrument():
	def __init__(self, ticker, date_range):
		self.id     = -1
		self.ticker = ticker

		if date_range is not None:
			self.date_range = date_range
		else:
			# If there was no specified time interval, presume that the
			# user intends to download historical price data from the 
			# past year. Notice that the end of the time interval is 
			# today, while the start is one year in the past.
			end = datetime.datetime.now().strftime("%Y-%m-%d")
			start = (datetime.datetime.now() - datetime.timedelta(days = 365)).strftime("%Y-%m-%d")
			self.date_range = {"start" : start, "end" : end}

		try:
			self.download_data(ticker, date_range['start'], date_range['end'])
		except ValueError:
			raise ValueError("Invalid ticker symbol specified or else there was not an internet connection available.")

		self.return_statistics        = {}
		self.risk_statistics          = {}
		self.risk_analysis_statistics = {}


	def download_data(self, ticker, start, end):
		data = web.DataReader(ticker, data_source = 'yahoo', start = start, end = end)
		self.data = data
		return 

	def calculate_return_statistics(self):
		statistics = {}

		# [1] Occasionally, values of zero are obtained as an asset price. In all likelihood, this
		#	  value is rubbish and cannot be trusted, as it implies that the asset has no value. 
		#     In these cases, we replace the reported asset price by the mean of all asset prices.
		closing_prices = self.data['Adj Close'].to_frame()
		# closing_prices[closing_prices == 0] = closing_prices.mean()

		#######################################################################
		#           P_t - P_{t-1}											  #
		#  [2] R_t = -------------   ,   change_t = log(P_t) - log(P_{t - 1}) #
		#             P_{t-1}                                                 #
		#								   					                  #
		#      [*] Expected Annual Return  = R_t * 252        			      #
		#      [*] Annualized Return (APR) = AVG(SUM(R_t per Year))           #
		#                                                                     #
		#######################################################################
		statistics["returns"] = closing_prices.pct_change().dropna()
		statistics["log_returns"] = closing_prices.apply(lambda x: np.log(x) - np.log(x.shift(1)) ).dropna()

		# [3] [*] For the expected return, we simply take the mean value of the calculated daily returns.
		#     [*] Multiply the average daily return by the length of the time series in order to
        #         obtain the expected return over the entire period.
		cummulative_return = statistics["returns"].iloc[::-1].sum().values[0] 

		statistics["expected_daily_return"]  = statistics["returns"].mean().values[0] 
		statistics["expected_total_return"]  = statistics["expected_daily_return"] * len(statistics["returns"])
		statistics["expected_annual_return"] = statistics["expected_daily_return"] * 252 
		statistics["APR"]                    = statistics["returns"].resample('Y').sum().mean().values[0]  
		statistics["APY"]                    = ((1 + cummulative_return)**(252 / len(statistics["returns"]) ) - 1 )

		self.return_statistics = statistics
		return 

	def calculate_risk_statistics(self):
		statistics = {}

		# [1] Retrieve Closing prices
		returns = self.return_statistics['returns']

		##############################################################
		#                 ____________________                       #
		#                |              _ 		    		         #
		#                |  SUM( R_t - R_t)				             #			 
		#  [2] σ_t   = \ |  -----------------   , Var_t = σ_t ^ 2    #
		#               \|  # trading days                           #
		#								___				     	     #
		#      [*] Annual Std = σ_t * \|252        		     		 #
		#      [*] Annual Var = σ_t^2 * 252                          #
		#                                                            #
		##############################################################
		# standrd deviation
		statistics["daily_std"]  = returns.std().values[0]
		statistics["total_std"]  = statistics["daily_std"] * np.sqrt(len(returns))
		statistics["annual_std"] = statistics["daily_std"] * np.sqrt(252)

		# variance
		statistics["daily_var"]  = statistics["daily_std"] ** 2
		statistics["total_var"]  = statistics["total_std"] ** 2
		statistics["annual_var"] = statistics["annual_std"] ** 2

		# statistics["annual_std"] = (returns.resample('Y').std() * np.sqrt(252)).mean().values[0] 

		self.risk_statistics = statistics
		return 

	def calculate_statistics(self):
		self.calculate_return_statistics()
		self.calculate_risk_statistics()
		return

	def risk_analysis(self):
		#########################################################
		##                Capital Asset Pricing Model          ##
		## ___________________________________________________ ##
		##       E[R_I] - R_RF  = α + β * (E[R_M] - R_RF)      ##
		##                                                     ##
		##       INPUT :  [*] E[R_I] = Expected Annual Return  ##
		##       OUTPUT : [*] α,β                              ##
		#########################################################

		start, end = self.date_range['start'], self.date_range['end']
		statistics = {}

		# [1] Retrieve Data for [*] Market               : S&P 500
		market = utils.market_info(self.date_range)
		
		# [2] Notations: [*] M  : Market
		#                [*] I  : Instrument
		#                [*] RF : Risk-Free
		#     Calculate return for the market

		statistics["return_I"]                  = self.return_statistics['returns']
		statistics["return_M"]                  = market["return_M"]
		statistics["expected_annual_return_M"]  = market["expected_annual_return_M"]
		statistics["annual_std_M"]              = market["annual_std_M"]
		statistics["return_RF"]                 = utils.risk_free_return(date_range = self.date_range)

		# [3] Caclulate correlation ρ_{I,M}
		statistics["correlation"] = statistics["return_I"].corrwith(statistics["return_M"]).values[0]

		# [4] Caclulate alpha, beta
		##########################################################################
		#					   σ_I			   							         #
		#		β = ρ_{Ι,M} * ----- ,  α = (E[R_I] - R_RF) - β*(E[R_M] - R_RF)	 #
		#					   σ_Μ			                  				     #
		#       [*] σ_Ι  = Annual Standard deviation of the instrument 	 		 #
		#       [*] σ_Μ  = Annual Standard deviation of the market 				 #
		#       [*] E[R_I] = Expected Annual Return of the instrument			 #
		#       [*] E[R_M] = Expected Annual Return of the Market 			     #
		##########################################################################
		R_RF    = statistics["return_RF"]
		corr_IM = statistics["correlation"]
		R_I     = self.return_statistics['expected_annual_return']
		STD_I   = self.risk_statistics['annual_std']
		
		R_M     = statistics["expected_annual_return_M"]
		STD_M   = statistics["annual_std_M"]

		statistics["beta"]  = corr_IM * STD_I / STD_M
		statistics["alpha"] = (R_I - R_RF) - statistics["beta"] * ( R_M - R_RF)


		# [4] Caclulate Sharpe Ratio SR, R squared R^2
		#############################################
		##          Annualised Sharpe Ratio (SR)   ##
		##_________________________________________##
		##              E[R_I] - R_RF              ##
		##         SR = --------------             ##
		##                   σ_I                   ##             
		#############################################
		###########################################################
		##                   R squared (R^2)                     ##
		##______________________________________________________ ##
		##										  ______		 ##
		##            SS_res         SUM(E[R_I] - E[R_I])        ##
		## R^2 = 1 - -------- = 1 - -----------------     = ρ^2  ##
		##								  ^	     ______		     ##
		##            SS_tot        SUM(E[R_I] - E[R_I])         ##             
		###########################################################
		statistics["sharpe_ratio"]  = (R_I - R_RF) / STD_I
		statistics["r_squared"]  = corr_IM ** 2

		self.risk_analysis_statistics = statistics

		return




	def explain_term(self, term):
		return












