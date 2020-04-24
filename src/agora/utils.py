import sys
import shlex
import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco
from tabulate import tabulate
from pandas_datareader import data as web
from functools import reduce


######################################################
##              Finance Related functions           ##
######################################################
portfolios = {'ret' : [], 'std' : [], 'sr': []}


def merge_instrument_returns(instrument_list, ticker_list ):
	returns = [ instrument.return_statistics['returns'] for instrument in instrument_list ]
	returns_df = reduce(lambda left,right: pd.merge(left,right, left_index = True, right_index = True), returns)
	returns_df.columns = ticker_list

	return returns_df


def risk_free_return(date_range):
	'''
		function:
			Retrieve Data for [*] Risk-free instrument : ^IRX or 3month Tbill
	'''
	start, end = date_range['start'], date_range['end']
	risk_free  = web.DataReader('^IRX', data_source = 'yahoo', start = start, end = end)['Adj Close'].to_frame()
	risk_free_return  = risk_free.pct_change().dropna().mean().values[0]
	return risk_free_return 

def market_info(date_range):
	'''
		function:
			Retrieve Data for [*] Market instrument : ^GSPC or S&P500
	'''
	start, end       = date_range['start'], date_range['end']
	market           = web.DataReader('^GSPC', data_source = 'yahoo', start = start, end = end)['Adj Close'].to_frame()
	returns_m        = market.pct_change().dropna()
	annual_return_m  = returns_m.mean().values[0] * 252 
	annul_std_m      = returns_m.std().values[0] * np.sqrt(252)
	result = {"returns_M" : returns_m, 
			  "expected_annual_return_M" : annual_return_m, 
			  "annual_std_M" : annul_std_m}
	return result


######################################################
##           User Interface (UI) Experience         ##
######################################################



def display(data):
	print(tabulate(data, headers = 'keys', tablefmt = 'psql'))
	return


def pprint(messages):
	spaces = len(max(messages, key = len)) + 2

	print("+" + "-"*spaces+ "+")
	for message in messages:
		print('|', end = '')
		print(" " + message + " ", end = '')
		print(" " * (spaces - 2 -len(message)), end = '')
		print('|')
	print("+" + "-"*spaces+ "+")
	return


def check_argv(limit, message):
	if len(sys.argv) != limit:
		print(message)
		return False

	return True

def run_command(cmd):
	"""
		Popen : executes a child program in a new process. A new pipe to the child should
		be created. Log files are directly connected to stderr stdout that are specified only
		under subprocess.Popen and not in the simplified subprocess.run
		 - universal_newlines = True since stderr and stdout are specified, so that
		   the respective log files are open in "text" format and not in "byte" format.

		Runs the command (cmd) and redirect output to the logfile.
	"""
	command_args = shlex.split(cmd)
	process = subprocess.Popen(command_args, universal_newlines = True)
	return 
	