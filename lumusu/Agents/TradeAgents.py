import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


# Variable Done checks if the buy signal has traded or not. If not
# traded then the reward will be 0 .

# Action 0 is short
# Action 1 is neutral
# Action 2 is long
class traderSignal:
	def __init__(self,input_size,action_size):
		self.input_size = input_size
		self.action_size = action_size
		self.memory = deque(maxlen = 1000)
		self.learning_rate = 0.01
		self.epsilon = 1.00
		self.epsilon_decay = 0.995
		self.epsilon_min = 0.1
		self.alpha = 0.001
		self.model = self._buildModel()

	def _buildModel(self):
		model = Sequential()
		model.add(Dense(64, input_dim = self.input_size , activation='relu'))
		model.add(Dense(32, activation='relu'))
		model.add(Dense(8,activation='relu'))
		model.add(Dense(self.action_size,activation='linear'))
		model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
		return model

	def remember(self,state,action,reward,next_state,done):
		self.memory.append((state, action, reward, next_state,done))

	def play(self,state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		profit_estimate = self.model.predict(state)
		return np.argmax(profit_estimate[0])

	def Learn_from_memory(self,batch_size = 32):
		minibatch = self.sample(self.memory,batch_size)
		for state,action,reward,next_state,done in minibatch:

			target = reward
			# Fval = action - 1
			if not done:
				target = reward + self.gamma * np.amax(self.model.predict(next_state))
			target_f = self.model.predict(state) # whether buy or not buy
			value = target_f[0][action]

			target_f[0][action] = value + self.alpha * ( target - value )
			self.model.fit(state, target_f, epochs=1, verbose=0)
 		
 		# reduce the value of epsilon for training	
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def load(self, name):
		self.model.load_weights(name)

	def save(self, name):
		self.model.save_weights(name)

# Trade Buy Order Agent

# done here checks if the value is less than low or not. If its low then done is false.
# If it is not low then done is true i.e the order will take place.

# The use of this agent is to estimate the value of low price using actions . It has seven actions
# to estimate and it will get reward on the basis of its estimate

class traderOrder:
	def __init__(self,input_size,action_size = 1):
		self.input_size = input_size
		self.action_size = action_size
		self.memory = deque(maxlen = 1000)
		self.learning_rate = 0.1
		self.epsilon = 1.00
		self.epsilon_decay = 0.995
		self.epsilon_min = 0.01
		self.alpha = 0.001
		self.model = self._buildModel()

	def _buildModel(self):
		model = Sequential()
		model.add(Dense(32 , input_dim = self.input_size , activation='relu'))
		model.add(Dense(16,activation='relu'))
		model.add(Dense(4,activation='relu'))
		model.add(Dense(self.action_size,activation='linear'))
		model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
		return model

	def remember(self,state,action,reward,next_state,done):
		self.memory.append((state, action, reward, next_state,done))

	def play(self,state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		Low = self.model.predict(state)
		return np.argmax(Low[0])

	def movingaverage(self,state):
		movingaverage = np.average(state)
		return movingaverage

	def Actionvalues(self,action):
		if (action == 0):
			return -12
		if (action == 1):
			return -7
		if (action == 2):
			return -3
		if (action == 3):
			return 0
		if (action == 4):
			return 3
		if (action == 5):
			return 7
		if (action == 6):
			return 12

	def Learn_from_memory(self,batch_size = 32):
		minibatch = self.sample(self.memory,batch_size)
		for state,action,reward,next_state,done in minibatch:
			target = reward
			if not done :
				target = 0 + np.amax(self.model.predict(next_state))

			target_f = self.model.predict(state)
			value = target_f[0][action]
			target_f[0][action] = value + self.gamma*(target - value)
			# target_f = target_f/np.sum(target_f[0])
			self.model.fit(state,target_f,epochs=1,verbose=0)
		

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def load(self, name):
		self.model.load_weights(name)

	def save(self, name):
		self.model.save_weights(name)

class traderOrder:
	def __init__(self,input_size,action_size = 1):
		self.input_size = input_size
		self.action_size = action_size
		self.memory = deque(maxlen = 1000)
		self.learning_rate = 0.1
		self.epsilon = 1.00
		self.epsilon_decay = 0.995
		self.epsilon_min = 0.01
		self.alpha = 0.001
		self.model = self._buildModel()

	def _buildModel(self):
		model = Sequential()
		model.add(Dense(32 , input_dim = self.input_size , activation='relu'))
		model.add(Dense(16,activation='relu'))
		model.add(Dense(4,activation='relu'))
		model.add(Dense(self.action_size,activation='linear'))
		model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
		return model

	def remember(self,state,action,reward,next_state,done):
		self.memory.append((state, action, reward, next_state,done))

	def play(self,state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		Low = self.model.predict(state)
		return np.argmax(Low[0])

	def movingaverage(self,state):
		movingaverage = np.average(state)
		return movingaverage

	def Actionvalues(self,action):
		if (action == 0):
			return -12
		if (action == 1):
			return -7
		if (action == 2):
			return -3
		if (action == 3):
			return 0
		if (action == 4):
			return 3
		if (action == 5):
			return 7
		if (action == 6):
			return 12

	def Learn_from_memory(self,batch_size = 32):
		minibatch = self.sample(self.memory,batch_size)
		for state,action,reward,next_state,done in minibatch:
			
			target = reward
			if not done :
				target = 0 + np.amax(self.model.predict(next_state))

			target_f = self.model.predict(state)
			value = target_f[0][action]
			target_f[0][action] = value + self.gamma*(target - value)
			# target_f = target_f/np.sum(target_f[0])
			self.model.fit(state,target_f,epochs=1,verbose=0)
		

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def load(self, name):
		self.model.load_weights(name)

	def save(self, name):
		self.model.save_weights(name)