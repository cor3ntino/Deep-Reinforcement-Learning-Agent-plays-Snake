from tkinter import Tk, Canvas, Label, Button
from constants import *
from snake import *
from map import *

class Interface():
    def __init__(self, agent=None, playable=True, trainable=False, convolution=False):

        self.fenetre = Tk()

        # We set parameters for visualization
        self.width = 400
        self.height = 400
        self.ratio = self.width / GAME_DIMENSION

        # RL agent
        self.agent = agent

        # Mode chosen to be played
        self.playable = playable
        self.trainable = trainable
        self.convolution = convolution

        if self.playable:
            # If playable, only 1 snake is played and shown
            self.nb_show = 1 # Number of snakes to show playing the same time
            self.nb_total = 1 # Total number of snakes to play at the same time
        else:
            # If not playable, 4 snakes are shown at the same time
            self.nb_show = 4
            self.ratio /= 2

            if self.trainable:
                # If trainable, a total of 32 snakes are played to improve converge and avoid being stuck in local minimum
                self.nb_total = 32
            else:
                self.nb_total = 4

        # Snakes and maps to be played are initialized
        self.snake = []
        self.map = []
        for _ in range(self.nb_total):
            s = Snake()
            self.snake.append(s)
            self.map.append(Map(s))

        if self.playable:
            # Activation of keyboard actions when playing mode 
            self.fenetre.bind("<Key>", self.on_key)
            self.fenetre.bind("<Right>", self.on_arraw_right)
            self.fenetre.bind("<Left>", self.on_arraw_left)
            self.fenetre.bind("<Up>", self.on_arraw_up)
            self.fenetre.bind("<Down>", self.on_arraw_down)

        if self.convolution:
            # In case of CNN chosen for neural network, we save the 3 previous frames of each snakes to enables CNN to takes account of snake's direction
            self.back_1 = [None] * self.nb_total
            self.back_2 = [None] * self.nb_total
            self.back_3 = [None] * self.nb_total

        if self.trainable:
            self.nb_events = 1 # To count the number of events for batch of training
            self.list_events = [] # Memory of last event to train model with RL agent

        # Initialization of the tkinter visual user interface
        self.canvas = Canvas(self.fenetre, width=self.width, height=self.height, background='white')

        label = Label(self.fenetre, text="")
        label.pack()
        label = Label(self.fenetre, text="GAME SNAKE", font="Times 20 bold")
        label.pack()
        label = Label(self.fenetre, text="")
        label.pack()
        p3 = Button(self.fenetre, text="PLAY", command=self.play)
        p3.pack()
        p4 = Button(self.fenetre, text="CLOSE", command=self.fenetre.quit)
        p4.pack()
        label = Label(self.fenetre, text="")
        label.pack()

        # Display environment before starting
        self.display()

        self.canvas.pack()
        self.fenetre.mainloop()

    def on_key(self, event):
        print(event.char)
    def on_arraw_right(self, event):
        self.snake[0].direction = [1, 0]
    def on_arraw_left(self, event):
        self.snake[0].direction = [-1, 0]
    def on_arraw_up(self, event):
        self.snake[0].direction = [0, -1]
    def on_arraw_down(self, event):
        self.snake[0].direction = [0, 1]

    def play(self):
        if self.playable: # To play by yourself
            for s, m in zip(self.snake, self.map):
                s.update()
                m.update()

            self.display()

            if self.snake[0].alive:
                self.fenetre.after(300, self.play)

        elif not self.trainable: # To play a pretrained model
            
            if not self.convolution:
                # DNN model
                for s, m in zip(self.snake, self.map):
                    state = m.scan()
                    action = self.agent.take_action(state)
                    s.update(decision=action)
                    m.update()
                
                self.display()

                for i in range(len(self.snake)):
                    if self.snake[i].alive == False:
                        self.snake[i] = Snake()
                        self.map[i] = Map(self.snake[i])

                self.fenetre.after(100, self.play)
            
            if self.convolution:
                # CNN model
                for num, [s, m] in enumerate(zip(self.snake, self.map)):
                    actual_state = m.give_matrice()
                    if type(self.back_3[num]) == np.ndarray:
                        state = [self.back_3[num], self.back_2[num], self.back_1[num], actual_state]
                        action = self.agent.take_action(state)
                    else:
                        # If we do not have the 3 previous frames (cold start)
                        action = 0
                    s.update(decision=action)
                    m.update()
                    
                    self.back_3[num], self.back_2[num], self.back_1[num] = self.back_2[num], self.back_1[num], actual_state

                self.display()

                for i in range(len(self.snake)):
                    if self.snake[i].alive == False:
                        self.snake[i] = Snake()
                        self.map[i] = Map(self.snake[i])
                        self.back_3[i], self.back_2[i], self.back_1[i] = None, None, None

                self.fenetre.after(100, self.play)

        else:
            # Train model
            if not self.convolution:
                # To train DNN with RL agent
                for s, m in zip(self.snake, self.map):
                    state = m.scan()
                    action = self.agent.take_action(state, trainable=True)
                    s.update(decision=action)
                    reward = m.update()
                    if s.alive:
                        next_state = m.scan()
                        self.list_events.append((state, action, reward, next_state))
                    else:
                        self.list_events.append((state, action, reward, None))
                
                self.display()

                for i in range(len(self.snake)):
                    if self.snake[i].alive == False:
                        self.snake[i] = Snake()
                        self.map[i] = Map(self.snake[i])

                if self.nb_events % 64 == 0: # Batch size of 64
                    stop_training = self.agent.train_model(self.list_events)
                    if stop_training:
                        self.trainable = False
                    self.list_events = []

                self.nb_events += 1

                self.fenetre.after(1, self.play)

            if self.convolution:
                # To train CNN with RL agent
                for num, [s, m] in enumerate(zip(self.snake, self.map)):
                    actual_state = m.give_matrice()
                    if type(self.back_3[num]) == np.ndarray:
                        state = [self.back_3[num], self.back_2[num], self.back_1[num], actual_state]
                        action = self.agent.take_action(state)
                    else:
                        # If we do not have the 3 previous frames (cold start)
                        action = 0
                    s.update(decision=action)
                    reward = m.update()

                    if type(self.back_3[num]) == np.ndarray:
                        if s.alive:
                            next_state = self.back_2[num], self.back_1[num], actual_state, m.give_matrice()
                            self.list_events.append((state, action, reward, next_state))
                        else:
                            self.list_events.append((state, action, reward, None))
                    
                    self.back_3[num], self.back_2[num], self.back_1[num] = self.back_2[num], self.back_1[num], actual_state

                self.display()

                for i in range(len(self.snake)):
                    if self.snake[i].alive == False:
                        self.snake[i] = Snake()
                        self.map[i] = Map(self.snake[i])
                        self.back_3[i], self.back_2[i], self.back_1[i] = None, None, None

                if self.nb_events % 64 == 0: # Batch size of 64
                    stop_training = self.agent.train_model(self.list_events)
                    if stop_training:
                        self.trainable = False
                    self.list_events = []

                self.nb_events += 1

                self.fenetre.after(1, self.play)

    def display(self):
        # Display user interface
        self.canvas.delete("all") # We start by removing objects of the previous frame
        
        s, m = self.snake[0], self.map[0]
        if self.nb_show == 1:
            # display walls
            self.canvas.create_rectangle(0, 0, self.ratio, self.width, fill='grey', outline='', width=0)
            self.canvas.create_rectangle(0, 0, self.width, self.ratio, fill='grey', outline='', width=0)
            self.canvas.create_rectangle(0, self.width - self.ratio, self.width, self.width, fill='grey', outline='', width=0)
            self.canvas.create_rectangle(self.width - self.ratio, 0, self.width, self.width, fill='grey', outline='', width=0)
            # display snake
            for k in range(1, len(s.body)):
                self.canvas.create_rectangle(s.body[k][0] * self.ratio, s.body[k][1] * self.ratio, (s.body[k][0] + 1) * self.ratio, (s.body[k][1] + 1) * self.ratio, fill='green', outline='', width=0)
            # display head of snake
            self.canvas.create_rectangle(s.body[0][0] * self.ratio, s.body[0][1] * self.ratio, (s.body[0][0] + 1) * self.ratio, (s.body[0][1] + 1) * self.ratio, fill='blue', outline='', width=0)
            # display food
            self.canvas.create_rectangle(m.food[0] * self.ratio, m.food[1] * self.ratio, (m.food[0] + 1) * self.ratio, (m.food[1] + 1) * self.ratio, fill='red', outline='', width=0)

        elif self.nb_show == 4:
            # display walls 1
            self.canvas.create_rectangle(0, 0, self.ratio, self.width / 2, fill='grey', outline='', width=0)
            self.canvas.create_rectangle(0, 0, self.width / 2, self.ratio, fill='grey', outline='', width=0)
            self.canvas.create_rectangle(0, self.width / 2 - self.ratio, self.width / 2, self.width / 2, fill='grey', outline='', width=0)
            self.canvas.create_rectangle(self.width / 2 - self.ratio, 0, self.width / 2, self.width / 2, fill='grey', outline='', width=0)
            # display walls 2
            self.canvas.create_rectangle(self.width / 2, 0, self.width / 2 + self.ratio, self.width / 2, fill='grey', outline='', width=0)
            self.canvas.create_rectangle(self.width / 2, 0, self.width, self.ratio, fill='grey', outline='', width=0)
            self.canvas.create_rectangle(0, self.width / 2 - self.ratio, self.width, self.width / 2, fill='grey', outline='', width=0)
            self.canvas.create_rectangle(self.width - self.ratio, 0, self.width, self.width / 2, fill='grey', outline='', width=0)
            # display walls 3
            self.canvas.create_rectangle(0, self.width / 2, self.ratio, self.width, fill='grey', outline='', width=0)
            self.canvas.create_rectangle(0, self.width / 2, self.width / 2, self.width / 2 + self.ratio, fill='grey', outline='', width=0)
            self.canvas.create_rectangle(0, self.width - self.ratio, self.width / 2, self.width, fill='grey', outline='', width=0)
            self.canvas.create_rectangle(self.width / 2 - self.ratio, self.width / 2, self.width / 2, self.width, fill='grey', outline='', width=0)
            # display walls 4
            self.canvas.create_rectangle(self.width / 2, self.width / 2, self.width / 2 + self.ratio, self.width, fill='grey', outline='', width=0)
            self.canvas.create_rectangle(self.width / 2, self.width / 2, self.width, self.width / 2 + self.ratio, fill='grey', outline='', width=0)
            self.canvas.create_rectangle(self.width / 2, self.width - self.ratio, self.width, self.width, fill='grey', outline='', width=0)
            self.canvas.create_rectangle(self.width - self.ratio, self.width / 2, self.width, self.width, fill='grey', outline='', width=0)
            
            # display 4 snakes
            for k in range(1, len(self.snake[0].body)):
                self.canvas.create_rectangle(self.snake[0].body[k][0] * self.ratio, self.snake[0].body[k][1] * self.ratio, (self.snake[0].body[k][0] + 1) * self.ratio, (self.snake[0].body[k][1] + 1) * self.ratio, fill='green')
            self.canvas.create_rectangle(self.snake[0].body[0][0] * self.ratio, self.snake[0].body[0][1] * self.ratio, (self.snake[0].body[0][0] + 1) * self.ratio, (self.snake[0].body[0][1] + 1) * self.ratio, fill='blue')
            for k in range(1, len(self.snake[1].body)):
                self.canvas.create_rectangle(self.height / 2 + self.snake[1].body[k][0] * self.ratio, self.snake[1].body[k][1] * self.ratio, self.height / 2 + (self.snake[1].body[k][0] + 1) * self.ratio, (self.snake[1].body[k][1] + 1) * self.ratio, fill='green')
            self.canvas.create_rectangle(self.height / 2 + self.snake[1].body[0][0] * self.ratio, self.snake[1].body[0][1] * self.ratio, self.height / 2 + (self.snake[1].body[0][0] + 1) * self.ratio, (self.snake[1].body[0][1] + 1) * self.ratio, fill='blue')
            for k in range(1, len(self.snake[2].body)):
                self.canvas.create_rectangle(self.snake[2].body[k][0] * self.ratio, self.height / 2 + self.snake[2].body[k][1] * self.ratio, (self.snake[2].body[k][0] + 1) * self.ratio, self.height / 2 + (self.snake[2].body[k][1] + 1) * self.ratio, fill='green')
            self.canvas.create_rectangle(self.snake[2].body[0][0] * self.ratio, self.height / 2 + self.snake[2].body[0][1] * self.ratio, (self.snake[2].body[0][0] + 1) * self.ratio, self.height / 2 + (self.snake[2].body[0][1] + 1) * self.ratio, fill='blue')
            for k in range(1, len(self.snake[3].body)):
                self.canvas.create_rectangle(self.height / 2 + self.snake[3].body[k][0] * self.ratio, self.height / 2 + self.snake[3].body[k][1] * self.ratio, self.height / 2 + (self.snake[3].body[k][0] + 1) * self.ratio, self.height / 2 + (self.snake[3].body[k][1] + 1) * self.ratio, fill='green')
            self.canvas.create_rectangle(self.height / 2 + self.snake[3].body[0][0] * self.ratio, self.height / 2 + self.snake[3].body[0][1] * self.ratio, self.height / 2 + (self.snake[3].body[0][0] + 1) * self.ratio, self.height / 2 + (self.snake[3].body[0][1] + 1) * self.ratio, fill='blue')

            # display 4 foods
            self.canvas.create_rectangle(self.map[0].food[0] * self.ratio, self.map[0].food[1] * self.ratio, (self.map[0].food[0] + 1) * self.ratio, (self.map[0].food[1] + 1) * self.ratio, fill='red')
            self.canvas.create_rectangle(self.height / 2 + self.map[1].food[0] * self.ratio, self.map[1].food[1] * self.ratio, self.height / 2 + (self.map[1].food[0] + 1) * self.ratio, (self.map[1].food[1] + 1) * self.ratio, fill='red')
            self.canvas.create_rectangle(self.map[2].food[0] * self.ratio, self.height / 2 + self.map[2].food[1] * self.ratio, (self.map[2].food[0] + 1) * self.ratio, self.height / 2 + (self.map[2].food[1] + 1) * self.ratio, fill='red')
            self.canvas.create_rectangle(self.height / 2 + self.map[3].food[0] * self.ratio, self.height / 2 + self.map[3].food[1] * self.ratio, self.height / 2 + (self.map[3].food[0] + 1) * self.ratio, self.height / 2 + (self.map[3].food[1] + 1) * self.ratio, fill='red')

            # display separation between 4 games
            self.canvas.create_line(0, self.height / 2, self.height, self.height / 2, fill='black')
            self.canvas.create_line(self.height / 2, 0, self.height / 2, self.height, fill='black')
