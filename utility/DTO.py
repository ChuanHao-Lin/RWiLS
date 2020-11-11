import numpy as np


class DTO(object):
    def __init__(self, a, b, N_h=20, rho=0.05, lambda_h=0.5, r_h=0.001):
        assert a < b

        self.N_h = N_h
        self.rho = rho
        self.lambda_h = lambda_h
        self.r_h = r_h

        self.q1 = [1 / self.N_h] * self.N_h

        self.DH = [(np.NINF, a)]
        for i in range(0, self.N_h - 2):
            self.DH.append((a + (b - a) / (self.N_h - 2) * i, a + (b - a) / (self.N_h - 2) * (i + 1)))
        self.DH.append((b, np.PINF))

        self.q = list()
        for i in self.q1:
            self.q.append((i + self.lambda_h) / (sum(self.q1) + self.N_h * self.lambda_h))

    def score_threshold(self, score):
        return score > self.threshold(score)

    def threshold(self, score):
        self.update(score)
        return self.threshold_t(score)

    def update(self,score):
        self.q1_update(score)
        self.q_update()

    def q1_update(self, score):
        for i in range(self.N_h):
            self.q1[i] = (1 - self.r_h) * self.q1[i]
        self.q1[self.s_star(score)] += self.r_h
        ### METHOD WITHOUT s_star FUNCTION ###
        #for i in range(self.N_h):
        #    if score >= self.DH[i][0] and score < self.DH[i][1]:
        #        self.q1[i] = (1 - self.r_h) * self.q1[i] + self.r_h
        #    else:
        #        self.q1[i] = (1 - self.r_h) * self.q1[i]

    def q_update(self):
        for i in range(self.N_h):
            self.q[i] = (self.q1[i] + self.lambda_h) / (sum(self.q1) + self.N_h * self.lambda_h)

    def threshold_t(self, score):
        # l_star instead of (l_star - 1) due to the fact that the bin label starts from 0 instead of 1
        return self.DH[self.s_star(score)][1] + (self.DH[-1][0] - self.DH[0][1]) / (self.N_h - 2) * self.l_star()
        #return self.l_star() + 1

    def reliability(self, score):
        return sum(self.q[self.l_star():]) / sum(self.q[self.s_star(score):])

    def s_star(self, score):
        for i in range(self.N_h):
            if score >= self.DH[i][0] and score < self.DH[i][1]:
                return i

    def l_star(self):
        for i in range(self.N_h):
            if sum(self.q[:i]) > 1 - self.rho:
                return i
        # No results. Set as negative zero so that threshold becomes INF
        return np.zeros(1, dtype=int)[0]
