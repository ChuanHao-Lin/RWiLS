import numpy as np
from .DTO import DTO


class aggregate(object):
    def __init__(self, a, b, N_h=20, rho=0.05, lambda_h=0.5, r_h=0.001, individual=False):
        self.scores = list()
        self.DTOs = list()

        self.individual_scores = None
        if individual:
            self.individual_scores = list()
        
        self.a = a
        self.b = b
        self.N_h = N_h
        self.rho = rho
        self.lambda_h = lambda_h
        self.r_h = r_h

    def default_DTO(self):
        return DTO(self.a, self.b, self.N_h, self.rho, self.lambda_h, self.r_h)

    def custom_DTO(self, a, b, N_h, rho, lambda_h, r_h):
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        if N_h is None:
            N_h = self.N_h
        if rho is None:
            rho = self.rho
        if lambda_h is None:
            lambda_h = self.lambda_h
        if r_h is None:
            r_h = self.r_h
        return DTO(a, b, N_h, rho, lambda_h, r_h)

    def change_DTO_setting(self):
        raise NotImplementedError

    def insert_new_data(self, data):
        raise NotImplementedError

    def get_scores(self):
        return self.scores

    def get_individual_scores(self):
        return self.individual_scores


##### 1-layer type #####
###        model
###  data =======> score
class aggregate_model(aggregate):
    def __init__(self, a, b, N_h=20, rho=0.05, lambda_h=0.5, r_h=0.001, individual=False):
        aggregate.__init__(self, a, b, N_h, rho, lambda_h, r_h, individual)
        self.models = list()

    def change_DTO_setting(self, idx=-1, a=None, b=None, N_h=None, rho=None, lambda_h=None, r_h=None):
        self.DTOs[idx] = self.custom_DTO(a, b, N_h, rho, lambda_h, r_h)

    def add_model(self, model):
        self.models.append(model)
        self.DTOs.append(self.default_DTO())
        if self.individual_scores is not None:
            self.individual_scores.append(list())

    def insert_new_data(self, data):
        agg_score = 0
        for i in range(len(self.models)):
            score = self.models[i](data)
            if self.individual_scores is not None:
                self.individual_scores[i].append(score)

            self.DTOs[i].update(score)
            agg_score += self.DTOs[i].reliability(score) * score
        self.scores.append(agg_score / len(self.models))


############## 2-layer type ##############
###        model        evaluation
###  data =======> out ============> score
class aggregate_model_evaluation(aggregate):
    def __init__(self, a, b, N_h=20, rho=0.05, lambda_h=0.5, r_h=0.001, individual=False):
        aggregate.__init__(self, a, b, N_h, rho, lambda_h, r_h, individual)
        self.models = list()
        self.evaluations = list()
                
    def change_DTO_setting(self, model_idx=-1, evaluation_idx=-1, a=None, b=None, N_h=None, rho=None, lambda_h=None, r_h=None):
        self.DTOs[model_idx][evaluation_idx] = self.custom_DTO(a, b, N_h, rho, lambda_h, r_h)
        
    def add_model(self, model):
        self.models.append(model)
        self.DTOs.append([self.default_DTO()] * len(self.evaluations))
        if self.individual_scores is not None:
            self.individual_scores.append([list()] * len(self.evaluations))
        
    def add_evaluation(self, evaluation):
        self.evaluations.append(evaluation)
        for i in range(len(self.models)):
            self.DTOs[i].append(self.default_DTO())
        if self.individual_scores is not None:
            for i in range(len(self.models)):
                self.individual_scores[i].append(list())
            

    def insert_new_data(self, data):
        agg_score = 0
        for i in range(len(self.models)):
            out = self.models[i](data)
            for j in range(len(self.evaluations)):
                score = self.evaluations[j](out)
                if self.individual_scores is not None:
                    self.individual_scores[i][j].append(score)
                    
                self.DTOs[i][j].update(score)
                agg_score += self.DTOs[i][j].reliability(score) * score
        self.scores.append(agg_score / len(self.models) / len(self.evaluations))
