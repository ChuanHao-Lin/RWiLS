import numpy as np


class sliding_window(object):
    def __init__(self, window_size, calculate=None):
        self.window = list()
        self.window_size = window_size
        # List of calculation function
        if calculate is None:
            self.calculate = list()
            self.scores = list()
        else:
            self.calculate = calculate
            self.scores = [list() for _ in calculate]

    def fill(self, data):
        try:
            self.window.append(data)
            if len(self.window) >= self.window_size:
                self.window = np.asarray(self.window)
                return False

            return True
        # already numpy (window is full)
        except:
            self.insert_new_data(data)
            return False

    def insert_new_data(self, data):
        if len(self.window) < self.window_size:
            return self.fill(data)
            
        while len(self.window) >= self.window_size:
            self.window = np.delete(self.window, 0, axis=0)
        self.window = np.vstack((self.window, data))

        for func, score in zip(self.calculate, self.scores):
            score.append(func(self.window))
 
        return True

    def get_window(self):
        return self.window

    def get_scores(self):
        return self.scores

    def add_calculate(self, func):
        self.calculate.append(func)
        self.scores.append(list())
