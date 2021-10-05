import os
import sys
import clr

assembly_path = os.path.join(os.getcwd(), r"bin")
sys.path.append(assembly_path)

clr.AddReference("Filpal.Electronics")
from System.Numerics import Complex
from Filpal.Electronics import CircuitDiagram
from Filpal.Electronics import HFDiagram
from Filpal.Electronics import TouchstoneFormat

# from s2p_analysis import S2P_ANALYSIS
# from test_analysis import S2P_ANALYSIS
from heuretic_analysis import HEURETIC
from test_s_para import TestSPara


class HFSS(object):
    def __init__(self, fpx_path, snp_path, fmin, fmax, fmid, passband_return_loss, stopband_insertion_loss,
                 num_ports, mclin_w_range, mclin_s_range):

        # Define *.fpx and *.snp file
        self.fpx_path = fpx_path
        self.snp_path = snp_path

        # S-Params general characteristics
        self.fmin = fmin
        self.fmax = fmax
        self.fmid = fmid

        self.passband_return_loss = passband_return_loss
        self.stopband_insertion_loss = stopband_insertion_loss

        # MCLIN general characteristics
        self.num_ports = num_ports
        self.mclin_w_range = mclin_w_range
        self.mclin_s_range = mclin_s_range

        # HFSS
        self.diagram = CircuitDiagram()
        self.diagram = CircuitDiagram.Load(self.fpx_path)
        self.s_params = []
        self.data = []

        # DQN
        self.action = []
        self.state = []
        self.params = []
        self.step_penalty = 0
        self.step_cnt = 0
        self.environment_reset()

    def environment_reset(self):
        self.params = []
        self.action = []
        self.step_penalty = 0
        self.step_cnt = 0
        self.diagram = CircuitDiagram()
        self.diagram = CircuitDiagram.Load(self.fpx_path)
        cnt = 0
        num_mclin = len(self.diagram.Symbols) - self.num_ports
        for symbol in self.diagram.Symbols:
            if cnt > self.num_ports - 1:
                component = symbol.Component
                for attribute in component.Attributes:
                    if attribute.Name == "W":
                        self.params.append(float(self.mclin_w_range[0] + (self.mclin_w_range[1] - self.mclin_w_range[0]) / 2))
                    if attribute.Name == "S":
                        self.params.append(float(self.mclin_s_range[0] + (self.mclin_s_range[1] - self.mclin_s_range[0]) / 2))
            cnt += 1

        mclin_action = [0.01e-3, -0.01e-3, 10e-6, -10e-6]
        for i in range(num_mclin):
            for a in mclin_action:
                self.action.append(a)

        self.s_params, self.data = self.environment_update()
        self.state, _, _ = self.s2p_evaluation()
        return self.state

    def step(self, action_index):
        delta = self.action[action_index]
        row_index = action_index // 4
        col_index = action_index % 4
        if col_index < 2:
            col_index = 0
        else:
            col_index = 1

        params_index = (row_index * 2) + col_index
        self.step_penalty += -0.04
        self.step_cnt += 1
        f = 0
        d = False
        if col_index == 0:
            if self.mclin_w_range[0] <= self.params[params_index] + delta < self.mclin_w_range[1]:
                expected = self.params[params_index] + delta
                # f += 100
            else:
                f += -100
                d = True
                expected = self.params[params_index] + delta
            self.params[params_index] = round(expected, 5)

        if col_index == 1:
            if self.mclin_s_range[0] <= self.params[params_index] + delta < self.mclin_s_range[1]:
                expected = self.params[params_index] + delta
                # f += 100
            else:
                f += -100
                d = True
                expected = self.params[params_index] + delta
            self.params[params_index] = round(expected, 5)

        if self.step_cnt >= 1000:
            d = True

        print(self.step_cnt)
        self.s_params, self.data = self.environment_update()
        self.state, done, reward = self.s2p_evaluation()
        done = True if d else done
        reward = reward + f + self.step_penalty
        return self.state, reward, done

    def environment_update(self):
        self.diagram = CircuitDiagram()
        self.diagram = CircuitDiagram.Load(self.fpx_path)
        cnt = 0
        mclin_data = []
        for symbol in self.diagram.Symbols:
            component = symbol.Component
            for i, attribute in enumerate(component.Attributes):
                if attribute.Name in ["W", "S"]:
                    v = Complex(self.params[cnt], 0)
                    attribute.Value = v
                    attribute.Expression = str(self.params[cnt])
                    component.SetFocus(i)
                    component.Calculate()
                    cnt += 1

        for symbol in self.diagram.Symbols:
            component = symbol.Component
            tmp = []
            for attribute in component.Attributes:
                if attribute.Name in ["Ze", "Zo", "EL"]:
                    tmp.append(float(attribute.Expression))
            if tmp:
                mclin_data.append(tmp)

        self.diagram.CountNode()
        self.diagram.FormMatrix()
        self.diagram.Plot()

        freq = []
        s11 = []
        s21 = []
        for datapoint in self.diagram.ChartParams["S11"]:
            data = str(datapoint).lstrip("(").rstrip(")").split(",")
            freq.append(int(data[0]))
            s11.append(float(data[1]))

        for datapoint in self.diagram.ChartParams["S21"]:
            data = str(datapoint).lstrip("(").rstrip(")").split(",")
            s21.append(float(data[1]))

        s_params = list(zip(freq, s11, s21))
        return s_params, mclin_data

    def s2p_evaluation(self):
        # s2p = S2P_ANALYSIS(self.s_params, self.fmin, self.fmax, self.fmid, self.passband_return_loss,
        #                    self.stopband_insertion_loss)
        # s_, d, r = s2p.get_state_reward()
        analysis = HEURETIC(self.data)
        s_, d, r = analysis.get_state_reward()
        return s_, d, r

    def export_spara_as_excel(self):
        snp_expected_path = r"Test/s2p/ParallelCoupledLine_2.s2p"
        for x in self.s_params:
            print(x)
        S = TestSPara(self.s_params, snp_expected_path)
        S.export()


if __name__ == "__main__":
    fps_path = r'Test\fpx\ParallelCoupledline_2.fpx'
    snp_path = r'Test\s2p\ParallelCoupledline.s2p'

    # Adjustables
    num_ports = 2
    f_min = 3e9
    f_max = 4e9
    f_mid = 3.5e9
    mclin_w_min = 2e-3
    mclin_w_max = 3e-3
    mclin_s_min = 10e-6
    mclin_s_max = 3000e-6
    passband_return_loss = -10
    stopband_insertion_loss = -50

    mclin_w_range = [mclin_w_min, mclin_w_max]
    mclin_s_range = [mclin_s_min, mclin_s_max]

    env = HFSS(fps_path, snp_path, f_min, f_max, f_mid, passband_return_loss, stopband_insertion_loss,
               num_ports, mclin_w_range, mclin_s_range)

    # observation_, rw, dn = env.step(2)
    observation_, rw, dn = env.step()
    print(observation_)
    print(rw)
    print(dn)
    # env.export_spara_as_excel()
