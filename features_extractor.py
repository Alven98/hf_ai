import os
import sys
import clr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

assembly_path = os.path.join(os.getcwd(), r"bin")
sys.path.append(assembly_path)

clr.AddReference("Filpal.Electronics")
from System.Numerics import Complex
from Filpal.Electronics import CircuitDiagram
from Filpal.Electronics import HFDiagram
from Filpal.Electronics import TouchstoneFormat


class HFSS(object):
    def __init__(self, fpx_path, params):

        # Define *.fpx
        self.fpx_path = fpx_path

        # HFSS
        self.diagram = CircuitDiagram()
        self.diagram = CircuitDiagram.Load(self.fpx_path)
        self.params = params
        self.cntr = 0

        # Data collected
        col = ['fo', 'ports', 'bandwidth', 'length', 'mclin', 'height', 'thickness', 'conductivity', 'dielectric', 'loss_tangent']
        self.db = pd.DataFrame(columns=col)
        self.fo_data = []
        self.bw_data = []
        self.port = []
        self.mclin = []
        self.height = []
        self.thickness = []
        self.conductivity = []
        self.dielectric = []
        self.loss_tangent = []

    def update_length(self):
        s11_data = []
        s21_data = []
        s11_plot = []
        s21_plot = []
        data_list = []

        self.diagram = CircuitDiagram()
        self.diagram = CircuitDiagram.Load(self.fpx_path)

        for symbol in self.diagram.Symbols:
            component = symbol.Component
            for i, attribute in enumerate(component.Attributes):
                if attribute.Name == "L":
                    v = Complex(self.params[self.cntr], 0)
                    attribute.Value = v
                    attribute.Expression = str(self.params[self.cntr])
                    component.SetFocus(i)
                    component.Calculate()

        fo = 1e9
        only_first = True
        mclin_cntr = 0
        port_cntr = 0
        for symbol in self.diagram.Symbols:
            component = symbol.Component
            done = False
            if "P" in component.Abbreviation:
                port_cntr += 1

            if "MCLIN" in component.Abbreviation:
                mclin_cntr += 1
                while not done and only_first:
                    val = Complex(fo, 0)
                    for attr in component.Attributes:
                        if attr.Name == "Freq":
                            attr.Value = val
                            attr.Expression = str(fo)
                    component.SetFocus(0)
                    component.Calculate()

                    obj_el = [attr.Value for attr in component.Attributes if attr.Name == "EL"]
                    el = float(str(obj_el[0]).strip('(').split(',')[0])

                    if self.cntr < 409:
                        if 89.99 < el < 90.7:
                            data_list.append(fo)
                            done = True
                            only_first = False

                    elif 409 <= self.cntr < 505:
                        if 89 < el < 90.9:
                            data_list.append(fo)
                            done = True
                            only_first = False

                    else:
                        if 88 < el < 93:
                            data_list.append(fo)
                            done = True
                            only_first = False

                    fo += 0.01e9

        fo -= 0.01e9
        min_frequency = float(fo - 1e9)
        max_frequency = float(fo + 1e9)

        self.diagram.Settings.MinFrequency = min_frequency
        self.diagram.Settings.MaxFrequency = max_frequency

        self.fo_data.append(data_list[0])
        self.diagram.CountNode()
        self.diagram.FormMatrix()
        self.diagram.Plot()

        for datapoint in self.diagram.ChartParams["S11"]:
            s11_data.append((float(str(datapoint).lstrip('(').split(',')[0]), float(str(datapoint).rstrip(')').split(',')[1])))
            s11_plot.append(float(str(datapoint).rstrip(')').split(',')[1]))
        for datapoint in self.diagram.ChartParams["S21"]:
            s21_data.append((float(str(datapoint).lstrip('(').split(',')[0]), float(str(datapoint).rstrip(')').split(',')[1])))
            s21_plot.append(float(str(datapoint).rstrip(')').split(',')[1]))

        flag = True
        fc1 = 0
        fc2 = 0
        for s in s21_data:
            if flag and s[1] > -3:
                fc1 = s[0]
                flag = False
            if not flag and s[1] < -3:
                fc2 = s[0]
                break

        self.bw_data.append(fc2 - fc1)
        self.port.append(port_cntr)
        self.mclin.append(mclin_cntr)
        self.height.append(self.diagram.Settings.MicrostripHeight)
        self.thickness.append(self.diagram.Settings.MicrostripThickness)
        self.conductivity.append(self.diagram.Settings.MicrostripConductivity)
        self.dielectric.append(self.diagram.Settings.MicrostripDielectric)
        self.loss_tangent.append(self.diagram.Settings.MicrostripTangent)

        self.cntr += 1

        if self.cntr == len(self.params):
            self.export_to_excel()

        print('Finish ' + str(self.cntr) + ' of ' + str(len(self.params)))

    def plot_s_parameters(self, min_freq, max_freq, s11_plot, s21_plot):
        x = np.linspace(min_freq, max_freq,  self.diagram.Settings.NumberOfPoints)
        plt.plot(x, s11_plot, label="s11")
        plt.plot(x, s21_plot, label="s21")
        plt.legend()
        plt.title('S-Parameters')
        plt.xlabel('Freq')
        plt.ylabel('dB')
        plt.grid()
        plt.show()

    def export_to_excel(self):
        self.db["fo"] = self.fo_data
        self.db["ports"] = self.port
        self.db["bandwidth"] = self.bw_data
        self.db["length"] = self.params
        self.db["mclin"] = self.mclin
        self.db["height"] = self.height
        self.db["thickness"] = self.thickness
        self.db["conductivity"] = self.conductivity
        self.db["dielectric"] = self.dielectric
        self.db["loss_tangent"] = self.loss_tangent

        dataset_path = r'datasets/data.xlsx'
        self.db.to_excel(dataset_path, index=None, header=True)


if __name__ == '__main__':
    fps_path = r'Test\fpx\ParallelCoupledline_4.fpx'
    params = np.arange(5e-3, 55e-3, 0.1e-3)
    hfss = HFSS(fps_path, params)

    for i in range(len(params)):
        hfss.update_length()


