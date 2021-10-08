import pandas as pd


class HEURETIC(object):
    def __init__(self, data, cf, el):
        self.data_db = pd.DataFrame(data, columns=["Ze", "Zo", "EL"])
        self.cf = cf
        self.el = el
        self.cf_min = self.cf - 0.1e-3
        self.cf_max = self.cf + 0.1e-3
        self.el_min = self.el - 0.01
        self.el_max = self.el + 0.01

    def width_analysis(self):
        """
        The get_state_reward for
        1. Construct State according to W
        2. Evaluation

        state (list): The state. Attributes:
            s[0] is the EL of MCLIN

        returns:
             a: To return the next state, reward and done status.
        """
        A1 = self.data_db.iloc[0]["EL"]

        state = [
            A1
        ]

        A = -1000 * (0 if self.el_min < abs(state[0]) < self.el_max else abs(self.el - state[0]) / (abs(self.el + state[0]) / 2))
        B = 100 * (1 if self.el_min < abs(state[0]) < self.el_max else 0)

        reward = (
            -1000 * (0 if self.el_min < abs(state[0]) < self.el_max else abs(self.el - state[0]) / (abs(self.el + state[0]) / 2))
            + 100 * (1 if self.el_min < abs(state[0]) < self.el_max else 0)
        )

        done = (
            True if reward > 80
            else False
        )

        return state, done, reward

    def space_analysis(self):
        """
        The get_state_reward for
        1. Construct State according to S
        2. Evaluation

        state (list): The state. Attributes:
            s[0] is the ratio of Zo/Ze of MCLIN

        returns:
             a: To return the next state, reward and done status.
        """
        A1 = self.data_db.iloc[0]["Ze"]
        A2 = self.data_db.iloc[0]["Zo"]

        state = [
            (A2 / A1)
        ]

        A = -100 * (0 if self.cf_min < abs(state[0]) < self.cf_max else abs(self.cf - state[0]) / (abs(self.cf + state[0]) / 2))
        C = 100 * (1 if self.cf_min < abs(state[0]) < self.cf_max else 0)

        reward = (
            -100 * (0 if self.cf_min < abs(state[0]) < self.cf_max else abs(self.cf - state[0]) / (abs(self.cf + state[0]) / 2))
            + 100 * (1 if self.cf_min < abs(state[0]) < self.cf_max else 0)
        )

        done = (
            True if reward > 80
            or self.cf_min < abs(state[0]) < self.cf_max
            else False
        )

        return state, done, reward


# if __name__ == '__main__':
#     from read_snp import ReadSNP
#     import os
#
#     fmin = 3
#     fmax = 4
#     fmid = (3 + 4) / 2
#
#     snp_path = os.path.join(os.getcwd(), r"Test/s2p/test1.s2p")
#     RS = ReadSNP(snp_path)
#     snp_db = RS.get_snp_db()
#
#     analysis = S2P_ANALYSIS(snp_db, fmin, fmax, fmid, -10, -50)
#     s_, r, d = analysis.get_state_reward()
#
#     print(s_)
#     print(r)
#     print(d)
