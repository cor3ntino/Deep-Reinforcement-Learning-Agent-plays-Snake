from interface import Interface

from DNN import Deep_Neural_Network
from CNN import Convolutional_Neural_Network
from Agent_Reinforcement_Learning import Agent_RL

from train import *

# %% Play by yourself the snake game
"""
Interface(playable=True)
"""

# %% Visualize one of the saved models
"""
#neural_net, convolution = Deep_Neural_Network(), False
#neural_net.load_mod()

neural_net, convolution = Convolutional_Neural_Network(), True
neural_net.load_mod()

agent = Agent_RL(model=neural_net)
Interface(neural_net=neural_net, agent=agent, playable=False, trainable=False, convolution=convolution)
"""
# %% Train Neural Network with Reinforcement Learning

#neural_net, convolution = Deep_Neural_Network(), False # Train with DNN
neural_net, convolution = Convolutional_Neural_Network(), True # Train with CNN
neural_net.load_mod()

agent = Agent_RL(model=neural_net)
tt(agent)
#Interface(neural_net=neural_net, agent=agent, playable=False, trainable=True, convolution=convolution)
