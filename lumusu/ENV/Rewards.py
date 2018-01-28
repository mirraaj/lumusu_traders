import random
import numpy as np 
import pandas as pd 
import keras

class Rewards:
	def __init__(self, lookback = 8):
		# self.state = state
		# self.next_state = state
		self.lookback = lookback

	def Actionvalues(self,action):
		if (action == 0):
			return -12.0
		if (action == 1):
			return -7.0
		if (action == 2):
			return -3.0
		if (action == 3):
			return 0.0
		if (action == 4):
			return 3.0
		if (action == 5):
			return 7.0
		if (action == 6):
			return 12.0

	def MovingAverage(self,state) :
		return np.average(state) 

	def BuySignalReward (self,action,state,next_state) :
		# Action 0 : Not Buy
		# Action 1 : Buy

		ma_state = self.MovingAverage(state)
		adj_next_state = next_state[-1]

		if action == 1 :
			diff = ma_state - adj_next_state
			reward = np.exp( diff ) * np.sign(diff)
			return reward
		else :
			diff = ma_state - adj_next_state
			reward =  - np.sign(diff) * np.exp( diff )
			return reward

	def BuyOrderReward (self,state,next_state,lowpriceEstimate) :
		# Action 0 = -12 %
		# Action 1 = -7  %
		# Action 2 = -3  %
		# Action 3 = 0   %
		# Action 4 = 3   %
		# Action 5 = 7   %
		# Action 6 = 12  %

		diff = lowpriceEstimate - next_state[-1]
		reward = np.sign(diff) * np.exp(-diff)
		return reward

	def SellSignalReward (self,action , state , next_state ) :
		# Action 0 : Hold
		# Action 1 : Sell

		ma_state = self.MovingAverage(state)
		adj_next_state = next_state[-1]

		if action == 0 :
			diff = ma_state - adj_next_state
			reward = np.exp( diff ) * np.sign(diff)
			return reward
		else :
			diff = ma_state - adj_next_state
			reward =  - np.sign(diff) * np.exp( diff )
			return reward
		
	def SellOrderReward (self,state,next_state,highpriceEstimate) :
		# Action 0 = -12 %
		# Action 1 = -7  %
		# Action 2 = -3  %
		# Action 3 = 0   %
		# Action 4 = 3   %
		# Action 5 = 7   %
		# Action 6 = 12  %

		diff = highpriceEstimate - next_state[-1]
		reward = np.sign(diff) * np.exp(diff)	
		return reward
