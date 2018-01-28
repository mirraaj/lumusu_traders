import lumusu.ENV.TradeGym as gym
import lumusu.Agents.TradeAgents as Agents
import lumusu.ENV.Rewards as Reward
import keras
import numpy as np 
import pandas as pd
import lumusu.utils as utils

EPISODES = 1000
lookback = 8
SignalActionSpace = 2
OrderActionSpace = 7
batch_size = 32	




if __name__ == '__main__':
	path = 'dataset/AAPL.csv'
	env = gym.make(path,lookback)
	# print env.data
	reward = Reward.Rewards()

	buy_signal = Agents.agent(lookback,SignalActionSpace,gamma=0.85)
	sell_signal = Agents.agent(lookback,SignalActionSpace,gamma=0.85)
	buy_order = Agents.agent(lookback,OrderActionSpace)
	sell_order = Agents.agent(lookback,OrderActionSpace)

	# profit = 0
	# bid_price = 0 

	reward_bs = 0
	reward_bo = 0
	reward_ss = 0
	reward_so = 0

	buy_price = 0
	sell_price = 0
	profit = 0
	avg_ret = 1
	profit_reward = 0.0
	days = 0
	for episode in range(EPISODES):
		days = 0
		profit = 0
		profit_reward = - np.inf
		buy_price = np.inf
		sell_price = -np.inf

		state = env.reset()
		next_state = env.step()
		buy_signal_action = 0
		
		while (not buy_signal_action):
			buy_signal_action , reward_bs ,adj_close , next_adj_close = utils.BuySignal(state,next_state,buy_signal)

			if (not buy_signal_action):
				buy_signal.remember(adj_close , buy_signal_action , reward_bs , next_adj_close, False)
				state = next_state
				next_state = env.step()	
				days += 1
		
		bo_action , reward_bo , buy_est , low_state ,low_nxt_state , = utils.BuyOrder(state,next_state,sell_signal)
		if (buy_est >=(low_nxt_state.values)[-1]) :
			buy_price = buy_est
		else:
			buy_price = (next_state['Close'].values)[-1] # The close price of that day	

		# buy_order.remember(low_state , bo_action , reward_bo, low_nxt_state )
		
		state = next_state
		next_state = env.step()
		days += 1
		# sell = env.copy()
		sell_sig_act = 0
		while(not sell_sig_act):
			sell_sig_act , reward_ss , sell_adj , sell_adj_next =utils.SellSignal(state,next_state,sell_signal)
			if (not sell_sig_act):
				sell_signal.remember(sell_adj,sell_sig_act,reward_ss,sell_adj_next)
				state = next_state
				next_state = env.step()
				days += 1

		so_act , reward_so , sell_est , high_state , next_high_state = utils.SellOrder(state,next_state,sell_order)
		
		# sell_order.remember(high_state , so_act ,reward_so , next_high_state)

		if sell_est <= (next_high_state.values)[-1]:
			sell_price = sell_est
		else:
			sell_price = (next_state['Close'].values)[-1]
		profit = ( sell_price - buy_price )/ buy_price
		profit_reward = 10*profit

		avg_ret = avg_ret * (1 + profit)

		buy_signal.remember(adj_close , buy_signal_action , reward_bs + profit_reward , next_adj_close)
		buy_order.remember(low_state , bo_action , reward_bo + profit_reward, low_nxt_state )	
		sell_signal.remember(sell_adj,sell_sig_act,reward_ss + profit_reward,sell_adj_next)
		sell_order.remember(high_state , so_act ,reward_so + profit_reward, next_high_state)

		print ("Episode : {}/{} , profit : {}  , days : {} , alpha : {:.2} , epsilon : {:.2}".format(episode,EPISODES,profit,days,\
			buy_signal.alpha,buy_signal.epsilon))
		
		if len(buy_signal.memory) > batch_size:
			buy_signal.replayMemory(batch_size)

		if len(sell_signal.memory) > batch_size:
			sell_signal.replayMemory(batch_size)

		if len(buy_order.memory) > batch_size:
			buy_order.replayMemory(batch_size)

		if len(sell_order.memory) > batch_size:
			sell_order.replayMemory(batch_size)				



# In this code the problem was :
# It was learing to maximize days
# Still giving negative profits
# Maybe it maximises the award by waiting
# Hence trying for a new profit funtion


