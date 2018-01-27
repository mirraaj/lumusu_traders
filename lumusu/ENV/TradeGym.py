import numpy as np 
import pandas as pd
import keras

# Store only ['Open', 'High', 'Low', 'Close', 'Adj Close'] in state
class load:
	def __init__(self,path):
		self.data = pd.read_csv(path)
		# self.data = self.data
		self.action_space = 3 # long , neutral and short
		self.lookback = 8 # how much back to look
		self.length = len(self.data)
		self.prev_action = 1 # Orignal Position is neutral (it notes buy or sell position)
		self.buy_price = np.inf 
		self.sell_price = np.inf
		self.transaction = 0.01
		self.wait = .01
		# self.start = random.randrange(self.length - 8) + 8
		# self.state = self.data.iloc[range(self.start - 8 ,self.start),:]
	def reset(self):
		self.prev_action = 1 # Orignal Postition is neutral
		self.start = random.randrange(self.start - 8) + 8
		self.state = self.data.iloc[range(self.start - 8 ,self.start),1:6]
		return self.state

	def step(self,action,price,buy_price = self.buy_price,sell_price = self.sell_price):
		self.buy_price = buy_price
		self.sell_price = sell_price

		if (self.prev_action == action):
			self.start = self.start + 1
			self.state = self.data.iloc[range(self.start - 8 ,self.start),1:6]
			if self.start == self.length:
				return self.state, -10 , True 
	
			return self.state , 0 , False

		elif (self.prev_action == 1) and (action == 0): 
			return self.state , -10 , True

		elif (self.prev_action == 1) and (action == 2):	
			self.prev_action = 2
			self.start = self.start + 1
			self.state = self.data.iloc[range(self.start - 8 ,self.start),1:6]
			if self.start == self.length:
				return self.state, -10 , True 

			reward = ( -self.state.iloc[-1,1] + self.state.iloc[-2,1]) / self.buy_price # -(high - bp) / bp
			return self.state,reward, False

		elif (self.prev_action == 2) and (action == 0) : #closing your position
			self.start = self.start + 1
			self.state = self.data.iloc[range(self.start - 8 ,self.start),1:6]
			reward = (1 - 2*self.transaction)( self.sell_price - self.buy_price ) / self.buy_price
			return self.state,reward, True

		elif (self.prev_action == 2) and (action == 1):	
			# self.prev_action = 2
			self.start = self.start + 1
			self.state = self.data.iloc[range(self.start - 8 ,self.start),1:6]
			if self.start == self.length:
				return self.state, -10 , True 

			reward = - self.wait * (self.state.iloc[-1,1] - self.buy_price) / self.buy_price 	
			return self.state , reward , False

		else :
			return self.state , -10 , True




		

		

