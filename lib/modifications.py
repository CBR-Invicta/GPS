from math import sinh, tanh
import pandas as pd
# from lib.visit_model_nn import VisitModelDescription


class modify:
    def __init__(self, visit_type: str, prev_proc: bool, genetic: bool):
        self.visit_type = visit_type
        self.prev_proc = prev_proc
        self.genetic = genetic

    def recalculate_predictions(self, predictions):
        if self.visit_type == "start_protocol":
            if self.prev_proc is True:
                return self.get_prot_prev_proc_modifications(predictions)
            else:
                return self.get_prot_wo_prev_proc_modifications(predictions)
        else:
            if self.prev_proc is True:
                return self.get_stim_prev_proc_modifications(predictions)
            else:
                return self.get_stim_wo_prev_proc_modifications(predictions)

    def get_prot_prev_proc_modifications(self, predictions):
        if self.genetic is True:
            predictions_new = [
                max(x * tanh(8 * (x - 2.083118) / 10.387512), 0)
                if x < 6
                else x
                * min((1 + (0.05) * sinh(((-2) + 4 * (x - 2.083118) / 10.387512))), 1.2)
                for x in predictions
            ]
        else:
            predictions_new = [
                max(x * tanh(10 * (x - 1.4495201) / 16.676577), 0)
                if x < 6
                else x
                * min((1 + (0.05) * sinh(((-2) + 4 * (x - 1.4495201) / 16.676577))), 1.2)
                for x in predictions
            ]
        return predictions_new

    def get_prot_wo_prev_proc_modifications(self, predictions):
        if self.genetic is True:
            predictions_new = [
                max(x * tanh(6 * (x - 2.8221145) / 10.855364), 0)
                if x < 6
                else x
                * min((1 + (0.05) * sinh(((-2) + 4 * (x - 2.8221145) / 10.855364))), 1.2)
                for x in predictions
            ]
        else:
            predictions_new = [
                max(x * tanh(50* (x - 2.5851831) / 8.383741), 0)
                if x < 6
                else x
                for x in predictions
            ]
        return predictions_new

    def get_stim_prev_proc_modifications(self, predictions):
        if self.genetic is True:
            predictions_new = [
                max(x * tanh(1500 * (x - 3.2011175) / 12.678201), 0) for x in predictions
            ]
        else:
            predictions_new = [
                max(x * tanh(40* (x - 1.7462413) / 11.63761), 0)
                if x < 6
                else x
                for x in predictions
            ]
        return predictions_new

    def get_stim_wo_prev_proc_modifications(self, predictions):
        if self.genetic is True:
            predictions_new = [
                max(x * tanh(40 * (x - 2.093592) / 20.314634), 0)
                if x < 6
                else x
                * min((1 + (0.05) * sinh(((-2) + 4 * (x - 2.093592) / 20.314634))), 1.2)
                for x in predictions
            ]
        else:
            predictions_new = [
                max(x * tanh(40* (x - 1.9592942) / 8.857515), 0)
                if x < 6
                else x
                for x in predictions
            ]
        return predictions_new
