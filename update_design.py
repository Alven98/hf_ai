import os
import sys
import clr
import numpy as np

assembly_path = os.path.join(os.getcwd(), r"bin")
sys.path.append(assembly_path)

clr.AddReference("Filpal.Electronics")
from System.Numerics import Complex
from Filpal.Electronics import CircuitDiagram
from Filpal.Electronics import HFDiagram
from Filpal.Electronics import TouchstoneFormat

from heuretic_analysis import EVALUATION


class HFSS(object):
    def __init__(self, fpx_path, snp_path,
                 mclin_w_range, mclin_s_range,
                 fo, bandwidth, length, mclin_ids, initial_guess,
                 tuning_parameter, return_loss, insertion_loss):

        # Define *.fpx and *.snp file
        self.fpx_path = fpx_path
        self.snp_path = snp_path

        # MCLIN general characteristics
        self.mclin_w_range = mclin_w_range
        self.mclin_s_range = mclin_s_range
        self.mclin_ids = mclin_ids
        self.mclin_id = self.mclin_ids[0]
        self.tuning_parameter = tuning_parameter
        self.params = initial_guess
        self.value_update = self.params[self.mclin_id][self.tuning_parameter]

        # Filter Requirement
        self.fo = fo
        self.bandwidth = bandwidth
        self.length = length
        self.return_loss = return_loss
        self.insertion_loss = insertion_loss

        # S-Parameters
        self.s11_data = []
        self.s21_data = []

        # HFSS Object
        self.diagram = CircuitDiagram()
        self.diagram = CircuitDiagram.Load(self.fpx_path)

        # DQN
        self.state = None
        self.action = [0.01e-3, -0.01e-3]
        self.step_penalty = 0
        self.step_cnt = 0

        # Reset Environmrnt
        self.environment_reset()

    def environment_reset(self):
        # TODO: Change initial guess with predicted parameters from NN
        self.step_penalty = 0
        self.step_cnt = 0
        self.s11_data = []
        self.s21_data = []

        self.diagram = CircuitDiagram()
        self.diagram = CircuitDiagram.Load(self.fpx_path)

        keys = list(self.params.keys())
        cnt = 0
        for symbol in self.diagram.Symbols:
            if symbol.Label == keys[cnt]:
                component = symbol.Component
                for i, attribute in enumerate(component.Attributes):
                    if attribute.Name == "Freq":
                        v = Complex(self.fo, 0)
                        attribute.Value = v
                        attribute.Expression = str(self.fo)
                    if attribute.Name == "W":
                        v = Complex(self.params[keys[cnt]]["W"], 0)
                        attribute.Value = v
                        attribute.Expression = str(self.params[keys[cnt]]["W"])
                    if attribute.Name == "L":
                        v = Complex(self.length, 0)
                        attribute.Value = v
                        attribute.Expression = str(self.length)
                    if attribute.Name == "S":
                        v = Complex(self.params[keys[cnt]]["S"], 0)
                        attribute.Value = v
                        attribute.Expression = str(self.params[keys[cnt]]["S"])

                    component.SetFocus(i)
                    component.Calculate()
                cnt += 1

        self.environment_update()
        self.state, _, _ = self.s2p_evaluation()
        return np.array(self.state)

    def step(self, action_index):
        delta = self.action[action_index]
        self.step_penalty += -0.04
        self.step_cnt += 1
        f = 0
        d = False
        expected = self.value_update

        if self.tuning_parameter == "W":
            if self.mclin_w_range[0] <= self.value_update + delta < self.mclin_w_range[1]:
                expected = self.value_update + delta

            else:
                f += -100
                d = True

        if self.tuning_parameter == "S":
            if self.mclin_s_range[0] <= self.value_update + delta < self.mclin_s_range[1]:
                expected = self.value_update + delta

            else:
                f += -100
                d = True

        self.value_update = round(expected, 5)
        for x in self.mclin_ids:
            self.params[x][self.tuning_parameter] = self.value_update
        self.environment_update()
        self.state, done, reward = self.s2p_evaluation()
        done = True if d else done
        reward = reward + f + self.step_penalty
        return np.array(self.state), reward, done

    def environment_update(self):
        self.s11_data = []
        self.s21_data = []

        self.diagram = CircuitDiagram()
        self.diagram = CircuitDiagram.Load(self.fpx_path)

        keys = list(self.params.keys())
        cnt = 0
        for symbol in self.diagram.Symbols:
            if symbol.Label == keys[cnt]:
                component = symbol.Component
                for i, attribute in enumerate(component.Attributes):
                    if attribute.Name == "Freq":
                        v = Complex(self.fo, 0)
                        attribute.Value = v
                        attribute.Expression = str(self.fo)
                    if attribute.Name == "W":
                        v = Complex(self.params[keys[cnt]]["W"], 0)
                        attribute.Value = v
                        attribute.Expression = str(self.params[keys[cnt]]["W"])
                    if attribute.Name == "L":
                        v = Complex(self.length, 0)
                        attribute.Value = v
                        attribute.Expression = str(self.length)
                    if attribute.Name == "S":
                        v = Complex(self.params[keys[cnt]]["S"], 0)
                        attribute.Value = v
                        attribute.Expression = str(self.params[keys[cnt]]["S"])

                    component.SetFocus(i)
                    component.Calculate()
                cnt += 1

        min_frequency = float(self.fo - 0.5e9)
        max_frequency = float(self.fo + 0.5e9)

        self.diagram.Settings.MinFrequency = min_frequency
        self.diagram.Settings.MaxFrequency = max_frequency

        self.diagram.CountNode()
        self.diagram.FormMatrix()
        self.diagram.Plot()

        for datapoint in self.diagram.ChartParams["S11"]:
            self.s11_data.append((float(str(datapoint).lstrip('(').split(',')[0]), float(str(datapoint).rstrip(')').split(',')[1])))
        for datapoint in self.diagram.ChartParams["S21"]:
            self.s21_data.append((float(str(datapoint).lstrip('(').split(',')[0]), float(str(datapoint).rstrip(')').split(',')[1])))

    def s2p_evaluation(self):
        analysis = EVALUATION(self.s11_data, self.s21_data, self.fo, self.bandwidth, self.return_loss, self.insertion_loss)
        s_, d, r = analysis.scathering_analysis()

        return s_, d, r

# if __name__ == "__main__":
#     fps_path = r'Test\fpx\ParallelCoupledline_2.fpx'
#     snp_path = r'Test\s2p\ParallelCoupledline.s2p'
#
#     # Adjustables
#     mclin_w_min = 2e-3
#     mclin_w_max = 3e-3
#
#     mclin_w_range = [mclin_w_min, mclin_w_max]
#
#     env = HFSS(fps_path, snp_path, mclin_w_range)
#
#     # observation_, rw, dn = env.step(2)
#     observation_, rw, dn = env.step(0)
#     print(observation_)
#     print(rw)
#     print(dn)
#     # env.export_spara_as_excel()
