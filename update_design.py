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

from heuretic_analysis import HEURETIC


class HFSS(object):
    def __init__(self, fpx_path, snp_path, mclin_w_range, mclin_s_range, coupling_factor, el, tuning_parameter, mclin):

        # Define *.fpx and *.snp file
        self.fpx_path = fpx_path
        self.snp_path = snp_path

        # MCLIN general characteristics
        self.mclin_w_range = mclin_w_range
        self.mclin_s_range = mclin_s_range
        self.cf = coupling_factor
        self.el = el
        self.mclin_id = "MCLIN" + str(mclin)
        self.params = None
        self.data = []
        self.tuning_parameter = tuning_parameter

        # HFSS
        self.diagram = CircuitDiagram()
        self.diagram = CircuitDiagram.Load(self.fpx_path)

        # DQN
        self.state = None
        self.action = [0.01e-3, -0.01e-3]
        self.step_penalty = 0
        self.step_cnt = 0
        self.environment_reset()

    def environment_reset(self):
        self.params = None
        self.step_penalty = 0
        self.step_cnt = 0

        self.diagram = CircuitDiagram()
        self.diagram = CircuitDiagram.Load(self.fpx_path)

        if self.tuning_parameter == "W":
            for symbol in self.diagram.Symbols:
                if symbol.Label == self.mclin_id:
                    component = symbol.Component
                    for attribute in component.Attributes:
                        if attribute.Name == "W":
                            self.params = float(self.mclin_w_range[0] + (self.mclin_w_range[1] - self.mclin_w_range[0]) / 2)

        if self.tuning_parameter == "S":
            for symbol in self.diagram.Symbols:
                if symbol.Label == self.mclin_id:
                    component = symbol.Component
                    for attribute in component.Attributes:
                        if attribute.Name == "S":
                            self.params = float(self.mclin_s_range[0] + (self.mclin_s_range[1] - self.mclin_s_range[0]) / 2)

        self.data = self.environment_update()
        self.state, _, _ = self.s2p_evaluation()
        return np.array(self.state)

    def step(self, action_index):
        delta = self.action[action_index]
        self.step_penalty += -0.04
        self.step_cnt += 1
        f = 0
        d = False
        expected = self.params

        if self.tuning_parameter == "W":
            if self.mclin_w_range[0] <= self.params + delta < self.mclin_w_range[1]:
                expected = self.params + delta

            else:
                f += -100
                d = True

        if self.tuning_parameter == "S":
            if self.mclin_s_range[0] <= self.params + delta < self.mclin_s_range[1]:
                expected = self.params + delta

            else:
                f += -100
                d = True

        self.params = round(expected, 5)
        self.data = self.environment_update()
        self.state, done, reward = self.s2p_evaluation()
        done = True if d else done
        reward = reward + f + self.step_penalty
        return np.array(self.state), reward, done

    def environment_update(self):
        self.diagram = CircuitDiagram()
        self.diagram = CircuitDiagram.Load(self.fpx_path)

        calulated_data = []
        if self.tuning_parameter == "W":
            for symbol in self.diagram.Symbols:
                if symbol.Label == self.mclin_id:
                    component = symbol.Component
                    for i, attribute in enumerate(component.Attributes):
                        if attribute.Name == "W":
                            v = Complex(self.params, 0)
                            attribute.Value = v
                            attribute.Expression = str(self.params)
                        component.SetFocus(i)
                        component.Calculate()

        if self.tuning_parameter == "S":
            for symbol in self.diagram.Symbols:
                if symbol.Label == self.mclin_id:
                    component = symbol.Component
                    for i, attribute in enumerate(component.Attributes):
                        if attribute.Name == "S":
                            v = Complex(self.params, 0)
                            attribute.Value = v
                            attribute.Expression = str(self.params)
                        component.SetFocus(i)
                        component.Calculate()

        for symbol in self.diagram.Symbols:
            if symbol.Label == self.mclin_id:
                component = symbol.Component
                tmp = []
                for attribute in component.Attributes:
                    if attribute.Name in ["Ze", "Zo", "EL"]:
                        tmp.append(float(attribute.Expression))
                if tmp:
                    calulated_data.append(tmp)

        self.diagram.CountNode()
        self.diagram.FormMatrix()
        self.diagram.Plot()

        return calulated_data

    def s2p_evaluation(self):
        analysis = HEURETIC(self.data, self.cf, self.el)
        s_, d, r = np.array([0, 0]), False, 0
        if self.tuning_parameter == "W":
            s_, d, r = analysis.width_analysis()
        if self.tuning_parameter == "S":
            s_, d, r = analysis.space_analysis()
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
