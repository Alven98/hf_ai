import pandas as pd


class HEURETIC(object):
    def __init__(self, data):
        self.data_db = pd.DataFrame(data, columns=["Ze", "Zo", "EL"])

    def get_state_reward(self):
        """
        The get_state_reward for
        1. Construct State
        2. Evaluation

        state (list): The state. Attributes:
            s[0] is the ratio of Zo/Ze of MCLIN1
            s[1] is the ratio of Zo/Ze of MCLIN2
            s[2] is the ratio of Zo/Ze of MCLIN3
            s[3] is the ratio of Zo/Ze of MCLIN4
            s[4] is the ratio of Zo/Ze of MCLIN5
            s[5] is the EL of MCLIN1
            s[6] is the EL of MCLIN2
            s[7] is the EL of MCLIN3
            s[8] is the EL of MCLIN4
            s[9] is the EL of MCLIN5

        returns:
             a: To return the next state, reward and done status.
        """
        A1 = self.data_db.iloc[0]["Ze"]
        A2 = self.data_db.iloc[0]["Zo"]
        A3 = self.data_db.iloc[0]["EL"]

        B1 = self.data_db.iloc[1]["Ze"]
        B2 = self.data_db.iloc[1]["Zo"]
        B3 = self.data_db.iloc[1]["EL"]

        C1 = self.data_db.iloc[2]["Ze"]
        C2 = self.data_db.iloc[2]["Zo"]
        C3 = self.data_db.iloc[2]["EL"]

        D1 = self.data_db.iloc[3]["Ze"]
        D2 = self.data_db.iloc[3]["Zo"]
        D3 = self.data_db.iloc[3]["EL"]

        E1 = self.data_db.iloc[4]["Ze"]
        E2 = self.data_db.iloc[4]["Zo"]
        E3 = self.data_db.iloc[4]["EL"]

        state = [
            (A2 / A1),
            (B2 / B1),
            (C2 / C1),
            (D2 / D1),
            (E2 / E1),
            A3,
            B3,
            C3,
            D3,
            E3
        ]
        # print(state)
        # reward = -100 * (0 if 0.55 < abs(state[0]) < 0.6 else abs(0.5894 - state[0]) / (abs(0.5894 + state[0]) / 2))
        # reward += -100 * (0 if 0.85 < abs(state[1]) < 0.9 else abs(0.8770 - state[1]) / (abs(0.8770 + state[1]) / 2))
        # reward += -100 * (0 if 0.90 < abs(state[2]) < 0.91 else abs(0.9013 - state[2]) / (abs(0.9013 + state[2]) / 2))
        # reward += -100 * (0 if 0.85 < abs(state[3]) < 0.9 else abs(0.8770 - state[3]) / (abs(0.8770 + state[3]) / 2))
        # reward += -100 * (0 if 0.55 < abs(state[4]) < 0.6 else abs(0.5894 - state[4]) / (abs(0.5894 + state[4]) / 2))
        # reward += -100 * (0 if 90 < abs(state[5]) < 90.7 else abs(90.6 - state[5]) / (abs(90.6 + state[5]) / 2))
        # reward += -100 * (0 if 90 < abs(state[6]) < 90.7 else abs(90.69 - state[6]) / (abs(90.69 + state[6]) / 2))
        # reward += -100 * (0 if 90 < abs(state[7]) < 90.7 else abs(90.69 - state[7]) / (abs(90.69 + state[7]) / 2))
        # reward += -100 * (0 if 90 < abs(state[8]) < 90.7 else abs(90.69 - state[8]) / (abs(90.69 + state[8]) / 2))
        # reward += -100 * (0 if 90 < abs(state[9]) < 90.7 else abs(90.6 - state[9]) / (abs(90.6 + state[9]) / 2))
        # reward += +10 * (1 if 0.55 < abs(state[0]) < 0.6 else 0)
        # reward += +10 * (1 if 0.85 < abs(state[1]) < 0.9 else 0)
        # reward += +10 * (1 if 0.90 < abs(state[2]) < 0.91 else 0)
        # reward += +10 * (1 if 0.85 < abs(state[3]) < 0.9 else 0)
        # reward += +10 * (1 if 0.55 < abs(state[4]) < 0.6 else 0)
        # reward += +10 * (1 if 90 < abs(state[5]) < 90.7 else 0)
        # reward += +10 * (1 if 90 < abs(state[6]) < 90.7 else 0)
        # reward += +10 * (1 if 90 < abs(state[7]) < 90.7 else 0)
        # reward += +10 * (1 if 90 < abs(state[8]) < 90.7 else 0)
        # reward += +10 * (1 if 90 < abs(state[9]) < 90.7 else 0)

        reward = (
            -100 * (0 if 0.55 < abs(state[0]) < 0.6 else abs(0.5894 - state[0]) / (abs(0.5894 + state[0]) / 2))
            - 100 * (0 if 0.85 < abs(state[1]) < 0.9 else abs(0.8770 - state[1]) / (abs(0.8770 + state[1]) / 2))
            - 100 * (0 if 0.90 < abs(state[2]) < 0.91 else abs(0.9013 - state[2]) / (abs(0.9013 + state[2]) / 2))
            - 100 * (0 if 0.85 < abs(state[3]) < 0.9 else abs(0.8770 - state[3]) / (abs(0.8770 + state[3]) / 2))
            - 100 * (0 if 0.55 < abs(state[4]) < 0.6 else abs(0.5894 - state[4]) / (abs(0.5894 + state[4]) / 2))
            - 100 * (0 if 90 < abs(state[5]) < 90.7 else abs(90.6 - state[5]) / (abs(90.6 + state[5]) / 2))
            - 100 * (0 if 90 < abs(state[6]) < 90.7 else abs(90.69 - state[6]) / (abs(90.69 + state[6]) / 2))
            - 100 * (0 if 90 < abs(state[7]) < 90.7 else abs(90.69 - state[7]) / (abs(90.69 + state[7]) / 2))
            - 100 * (0 if 90 < abs(state[8]) < 90.7 else abs(90.69 - state[8]) / (abs(90.69 + state[8]) / 2))
            - 100 * (0 if 90 < abs(state[9]) < 90.7 else abs(90.6 - state[9]) / (abs(90.6 + state[9]) / 2))
            + 10 * (1 if 0.55 < abs(state[0]) < 0.6 else 0)
            + 10 * (1 if 0.85 < abs(state[1]) < 0.9 else 0)
            + 10 * (1 if 0.90 < abs(state[2]) < 0.91 else 0)
            + 10 * (1 if 0.85 < abs(state[3]) < 0.9 else 0)
            + 10 * (1 if 0.55 < abs(state[4]) < 0.6 else 0)
            + 10 * (1 if 90 < abs(state[5]) < 90.7 else 0)
            + 10 * (1 if 90 < abs(state[6]) < 90.7 else 0)
            + 10 * (1 if 90 < abs(state[7]) < 90.7 else 0)
            + 10 * (1 if 90 < abs(state[8]) < 90.7 else 0)
            + 10 * (1 if 90 < abs(state[9]) < 90.7 else 0)
        )

        done = (
            True if reward > 80
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
