import numpy as np 
import keras 
import pandas as pd 

import ENV.Rewards as Reward
import Agents.TradeAgents as Agents

Reward = Reward.Rewards()

def Actionvalues(action):
	if (action == 0):
		return -12.0/100
	if (action == 1):
		return -7.0/100
	if (action == 2):
		return -3.0/100
	if (action == 3):
		return 0.0/100
	if (action == 4):
		return 3.0/100
	if (action == 5):
		return 7.0/100
	if (action == 6):
		return 12.0/100
def MovingAvg(state):
	return np.average(state)

def BuySignal(state,next_state,BuySignal):
	reward = 0
	adj_close = state['Adj Close']
	action = BuySignal.play(adj_close)	
	next_adj_close = next_state['Adj Close']
	reward = Reward.BuySignalReward( action , adj_close.values , next_adj_close.values )
	return action,reward , adj_close , next_adj_close

def BuyOrder(state,next_state,BuyOrder) :
	reward = 0
	low_price = state['Low']
	next_low_price = next_state['Low']
	action = BuyOrder.play(low_price)
	priceEstimate = MovingAvg(low_price)*( 1 + Actionvalues(action))
	reward = Reward.BuyOrderReward(low_price.values,next_low_price.values,priceEstimate)
	return action,reward,priceEstimate,low_price,next_low_price

def SellSignal(state,next_state,SellSignal):
	reward = 0
	adj_close = state['Adj Close']
	action = SellSignal.play(adj_close)	
	next_adj_close = next_state['Adj Close']
	reward = Reward.SellSignalReward( action , adj_close.values, next_adj_close.values )
	return action,reward , adj_close , next_adj_close

def SellOrder(state,next_state,SellOrder) :
	reward = 0
	high_price = state['High']
	next_high_price = next_state['High']
	action = SellOrder.play(high_price)
	priceEstimate = MovingAvg(high_price)*( 1 + Actionvalues(action))
	reward = Reward.SellOrderReward(high_price.values,next_high_price.values,priceEstimate)
	return action,reward,priceEstimate,high_price,next_high_price
