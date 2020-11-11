import numpy as np
from scipy.special import loggamma, digamma
from dirichlet import dirichlet
from .util import log_nml_c_dirichlet

err_state = np.nan

class scaw(object):
    def __init__(self, err_probs=[0.05, 0.05, 0.05], cut=True, counter=None):
        self.err_probs = err_probs
        self.cut = cut
        self.d_orders = (1, 2)
        self.scores = [list(), list(), list()]
        self.change_points = [list(), list(), list()]
        self.window_sizes = list()
        if counter:
            self.counter = counter
        else:
            self.timestamp = 0
            self.counter = self.default_counter

    def log_nml_c(self):
        raise NotImplementedError

    def NML(self):
        raise NotImplementedError

    def change_score(self, window, h):
        raise NotImplementedError

    def DMDL(self, window, h, order=0):

        if order == 1:
            return self.change_score(window, h + 1) - self.change_score(window, h)

        if order == 2:
            return self.change_score(window, h + 1) - 2 * self.change_score(window, h) + self.change_score(window, h - 1)
            
        return self.change_score(window, h)

    def threshold(self, window_size, dim=1, order=0):
        if window_size <= 1:
            return 0

        if order == 0:
            return (2 + dim / 2 + self.err_probs[0]) * np.log(window_size) - np.log(self.err_probs[0])
        elif order == 1:
            return dim * np.log(window_size / 2) - np.log(self.err_probs[1])
        elif order == 2:
            return 2 * (dim * np.log(window_size / 2) - np.log(self.err_probs[2]))

    def best_cut(self, window, order=0):
        max_score = 0
        max_cut = 0
        for i in range(3, len(window) - 2):
#        for i in range(2, len(window) - 1):
#        for i in range(1, len(window)):
            score = self.DMDL(window, i, order)
            if score > max_score: # the error (NaN) will be automatically covered by 0 (default)
                max_score = score
                max_cut = i
        return max_score, window[max_cut:]

    def get_scores(self, order=None):
        try:
            return self.scores[order]
        except:
            return self.scores

    def get_change_points(self, order=None):
        try:
            return self.change_points[order]
        except:
            return self.change_points

    def default_counter(self):
        self.timestamp += 1
        return self.timestamp

    def get_window_sizes(self):
        return self.window_sizes


class scaw1(scaw):
    def __init__(self, err_probs=[0.05, 0.05, 0.05], cut=True, counter=None):
        scaw.__init__(self, err_probs, cut, counter)
        self.window = list()

    def insert_new_data(self, data):
        self.window.append(data)
        self.window_sizes.append(len(self.window))
        t = self.counter()

        for i in self.d_orders:
            score, _ = self.best_cut(self.window, i)
            self.scores[i].append(score)
            if score > self.threshold(len(self.window), len(data), i):
                self.change_points[i].append(t)
        # Seperate 0 with 1, 2
        score, max_cut = self.best_cut(self.window)
        self.scores[0].append(score)

        if score > self.threshold(len(self.window), len(data)):
            self.change_points[0].append(t)
            if self.cut:
                self.window = max_cut
            else:
                self.window = list()


class scaw2(scaw):
    def __init__(self, max_bucket_size, err_probs=[0.05, 0.05, 0.05], cut=True, counter=None):
        scaw.__init__(self, err_probs, cut, counter)
        self.bucket = list()
        self.bucket_sizes = list()
        self.max_bucket_size = max_bucket_size

    def insert_new_data(self, data):
        self.bucket.append(data)
        self.bucket_sizes.append(1)
        t = self.counter()
            
        while len(self.bucket_sizes) > self.max_bucket_size:
            for i in range(1, len(self.bucket_sizes)):
                if self.bucket_sizes[i - 1] == self.bucket_sizes[i]:
                    self.bucket[i - 1] = [ sum(col) / len(col) for col in zip(*self.bucket[i-1:i+1]) ]
                    self.bucket_sizes[i - 1] *= 2
                    self.bucket.pop(i)
                    self.bucket_sizes.pop(i)
                    break

        for i in self.d_orders:
            score, _ = self.best_cut(self.bucket, i)
            self.scores[i].append(score)
            if score > self.threshold(len(self.bucket), len(data), i):
                self.change_points[i].append(t)
        # Seperate 0 with 1, 2
        score, max_cut = self.best_cut(self.bucket)
        self.scores[0].append(score)

        self.window_sizes.append(len(self.bucket_sizes))

        if score > self.threshold(len(self.bucket), len(data)):
            self.change_points[0].append(t)
            if self.cut:
                self.bucket = max_cut
                self.bucket_sizes = self.bucket_sizes[-len(self.bucket):]
            else:
                self.bucket = list()
                self.bucket_sizes = list()


class scaw_dirichlet(scaw):
    def __init__(self, estimate_method="meanprecision"):
        self.estimate_method = estimate_method # meanprecision fixedpoint

    def log_nml_c(self, dim, window_size):
        return log_nml_c_dirichlet(dim, window_size)

    def NML(self, alpha, window_size, dimension):
        alpha_0 = sum(alpha)
        return window_size * (sum(loggamma(alpha)) - loggamma(alpha_0) - sum((alpha - 1) * (digamma(alpha) - digamma(alpha_0)))) + self.log_nml_c(dimension, window_size)

    def change_score(self, window, h):
        window = np.array(window)
        window_size, dim = window.shape

        try:
            before = dirichlet.mle(window[:h], method=self.estimate_method)
            after = dirichlet.mle(window[h:], method=self.estimate_method)
            alpha = dirichlet.mle(window, method=self.estimate_method)
        except:
            return err_state

        return self.NML(alpha, window_size, dim) - self.NML(before, h, dim) - self.NML(after, window_size - h, dim)


class scaw_gaussian(scaw):
    def __init__(self):
        pass

    def log_nml_c(self, k):
        return k / 2 * (np.log(k / 2) - 1) - loggamma((k - 1) / 2)

    def NML(self, std, window_size):
        return window_size * np.log(std) + self.log_nml_c(window_size)

    def multi_std(self, M):
        if len(M.shape) == 1 or any([i == 1 for i in M.shape]):
            return np.std(M)

        print(M.shape)
        return np.sqrt(np.linalg.norm(np.linalg.det(np.cov(M.T))))

    def change_score(self, window, h):
        window_size = len(window)
        window = np.array(window)
        
        try:
            std_all = self.multi_std(window)
            std_before = self.multi_std(window[:h])
            std_after = self.multi_std(window[h:])
        except:
            return err_state

        if std_before == 0 or std_after == 0:
            return err_state

        return self.NML(std_all, window_size) - self.NML(std_before, h) - self.NML(std_after, window_size - h)


class scaw1_dirichlet(scaw1, scaw_dirichlet):
    def __init__(self, err_probs=[0.05, 0.05, 0.05], cut=True, estimate_method="meanprecision", counter=None):
        scaw1.__init__(self, err_probs, cut, counter)
        scaw_dirichlet.__init__(self, estimate_method)


class scaw2_dirichlet(scaw2, scaw_dirichlet):
    def __init__(self, max_bucket_size, err_probs=[0.05, 0.05, 0.05], cut=True, estimate_method="meanprecision", counter=None):
        scaw2.__init__(self, max_bucket_size, err_probs, cut, counter)
        scaw_dirichlet.__init__(self, estimate_method)


class scaw1_gaussian(scaw1, scaw_gaussian):
    def __init__(self, err_probs=[0.05, 0.05, 0.05], cut=True, counter=None):
        scaw1.__init__(self, err_probs, cut, counter)
        scaw_gaussian.__init__(self)


class scaw2_gaussian(scaw2, scaw_gaussian):
    def __init__(self, max_bucket_size, err_probs=[0.05, 0.05, 0.05], cut=True, counter=None):
        scaw2.__init__(self, max_bucket_size, err_probs, cut, counter)
        scaw_gaussian.__init__(self)
