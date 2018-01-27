import lumusu.ENV.TradeGym as gym
import lumusu.Agents.TradeAgents as Agent

import keras
import numpy as np 
import pandas as pd

if __name__ == '__main__':
	path = 'dataset/AAPL.csv'
	env = gym.make(path)
	# print env.data
	env.reset()
	print env.buy_price
	_ , _ , _ = env.step(1 , buy_price = 10)
	print env.buy_price
	print env.sell_price
	_ , _ , _ = env.step(1 , sell_price = 12)
	print env.sell_price
	print env.buy_price
	env.reset()
	print env.sell_price
	print env.buy_price