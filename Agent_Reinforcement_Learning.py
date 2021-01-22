import numpy as np

import keras
from keras.optimizers import Adam

class Agent_RL():
    def __init__(self, model):
        
        self.num_actions = 3
        self.epsilon = 1.
        self.discount = 0.99
        
        self.model = model
    
    def take_action(self, state, trainable=False):
        if np.random.rand() < self.epsilon and trainable:
            action = np.random.randint(self.num_actions)
        else:
            action = self.argmax(self.model.return_value_NN(np.array([state]))[0])
        return(action)
    
    def train_model(self, list_events):
        inputs = []
        targets = []

        for state, decision, reward, next_state in list_events:
            inputs.append(state)
            target = self.model.return_value_NN(np.array([state]))[0]
            if next_state is None:
                target[decision] = reward
                targets.append(target)
            else:
                target[decision] = reward + self.discount * np.max(self.model.return_value_NN(np.array([next_state])))
                targets.append(target)
        
        print('Loss: ', self.model.model.train_on_batch(np.array(inputs), np.array(targets)))
        print("Epsilon: ", self.epsilon, '\n')

        """
        if self.epsilon < 0.01:
            # We save model when epsilon < 1 %
            pourcentage = str(0.1)
            self.model.save_mod(pourcentage)
            return(True)
        """
        if self.count_train == 2000:
            self.model.save_mod('5000_3000_2000_iterations_1_pourcents')
            return(True)
        # We decrease epsilon as long as the model learn in order to make it more and more greddy
        self.epsilon *= 0.995
        return(False)
        ## Perform an update of action-values
        ## Following the Q-learning method
        #self.q[self.prev_state, self.prev_action] += self.step_size * (reward + self.discount * current_q[np.argmax(current_q)] - self.q[self.prev_state, self.prev_action])
        
        ## Following the Expected Sarsa method
        #idx_greedy = np.where(self.q[state, :] == current_q[np.argmax(current_q)])[0]
        #pi = (self.epsilon / self.num_actions) * np.ones(self.q[state, :].shape)
        #pi[idx_greedy] += (1 - self.epsilon) / len(idx_greedy)
        #estim = np.matmul(self.q[state, :], np.transpose(pi))
        #self.q[self.prev_state, self.prev_action] += self.step_size * (reward + self.discount * estim - self.q[self.prev_state, self.prev_action])
        
    def argmax(self, values):
        """
        argmax with random tie-breaking
        """

        top = float("-inf")
        ties = []

        for i in range(len(values)):
            if values[i] > top:
                top = values[i]
                ties = []

            if values[i] == top:
                ties.append(i)

        return np.random.choice(ties)