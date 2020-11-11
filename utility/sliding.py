import numpy as np
from dirichlet import dirichlet
from .util import kl_dirichlet, nml_code_length


estimate_method = "fixedpoint" # meanprecision fixedpoint


def dirichlet_estimation_all(window, estimate_method=estimate_method):
    return dirichlet.mle(window, method=estimate_method)


def dirichlet_estimation_half(window, estimate_method=estimate_method):
    window_size = len(window)

    before = dirichlet.mle(window[:window_size//2], method=estimate_method)
    after = dirichlet.mle(window[window_size//2:], method=estimate_method)
    return before, after


def kl_score(window, estimate_method=estimate_method):
    before, after = dirichlet_estimation_half(window, estimate_method)
    return kl_dirichlet(after, before)


def smdl_score(window, estimate_method=estimate_method):
    window_size = len(window)

    before, after = dirichlet_estimation_half(window, estimate_method)
    alpha = dirichlet_estimation_all(window, estimate_method)
    return nml_code_length(alpha, window_size) - nml_code_length(before, window_size//2) - nml_code_length(after, window_size - window_size//2)


class sliding_window(object):
    def __init__(self, window_size, preprocess, estimate_method=estimate_method):
        self.window = list()
        self.window_size = window_size
        self.preprocess = preprocess
        self.estimate_method = estimate_method
        self.kl_scores = list()
        self.smdl_scores = list()

    def fill(self, g):
        try:
            self.window.append(self.preprocess(g))
            if len(self.window) >= self.window_size:
                self.window = np.asarray(self.window)
                return False

            return True
        # already numpy (full)
        except:
            self.insert_new_data(g)
            return False

    def insert_new_data(self, g):
        while len(self.window) >= self.window_size:
            self.window = np.delete(self.window, 0, axis=0)
        self.window = np.vstack((self.window, self.preprocess(g)))
 
        return self.window

    def calculate_scores(self):
        self.calculate_kl()
        self.calculate_smdl()

    def calculate_kl(self):
        self.kl_scores.append(kl_score(self.window, self.estimate_method))

    def calculate_smdl(self):
        self.smdl_scores.append(smdl_score(self.window, self.estimate_method))

    def get_kl_scores(self):
        return self.kl_scores

    def get_smdl_scores(self):
        return self.smdl_scores

    def get_scores(self):
        return self.get_kl_scores(), self.get_smdl_scores()