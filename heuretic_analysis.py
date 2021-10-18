import pandas as pd


class EVALUATION(object):
    def __init__(self, s11, s21, fo, bw, return_loss, insertion_loss):

        # Initialize s11, s21 database
        self.s11_data_db = pd.DataFrame(s11, columns=["f", "dB"])
        self.s21_data_db = pd.DataFrame(s21, columns=["f", "dB"])

        # Policy
        self.fo = fo
        self.fc1 = fo - (bw / 2)
        self.fc2 = fo + (bw / 2)
        self.bw = bw
        self.return_loss = return_loss
        self.insertion_loss = insertion_loss

    def scathering_analysis(self):
        """
        The get_state_reward for
        1. Construct State according to tuning parameters
        2. Evaluation

        state (list): The state. Attributes:
            s[0] is the first cutoff frequency
            s[1] is the second cutoff frequency
            s[2] is the bandwidth of S-P

        returns:
             a: To return the next state, reward and done status.
        """
        s21_passband = self.s21_data_db.loc[(self.s21_data_db['dB'] > -5)].values
        fc1 = s21_passband[0][0]
        fc2 = s21_passband[-1][0]
        actual_bw = fc2 - fc1

        passes = 0
        s11_passband = self.s11_data_db.loc[(self.s11_data_db['f'] >= fc1) & (self.s11_data_db['f'] <= fc2)].values
        for p in s11_passband:
            if p[1] <= self.return_loss:
                passes += 1
        pass_rate = passes / len(s11_passband)

        state = [
            fc1,
            fc2,
            actual_bw,
            pass_rate
        ]

        A = -100 * 0.6 * (0 if self.fc1 - 0.05e9 < state[0] < self.fc1 + 0.05e9 else abs(self.fc1 - state[0]) / (abs(self.fc1 + state[0]) / 2))
        B = -100 * 0.6 * (0 if self.fc2 - 0.05e9 < state[1] < self.fc2 + 0.05e9 else abs(self.fc2 - state[1]) / (abs(self.fc2 + state[0]) / 2))
        C = -100 * 0.4 * (0 if self.bw - 0.05e9 < state[2] < self.bw + 0.05e9 else abs(self.bw - state[2]) / (abs(self.bw + state[0]) / 2))
        D = -100 * 0.7 * (1 - state[3])
        E = 50 * (1 if self.fc1 - 0.05e9 < state[0] < self.fc1 + 0.05e9 else 0)
        F = 50 * (1 if self.fc2 - 0.05e9 < state[1] < self.fc2 + 0.05e9 else 0)
        G = 100 * (1 if self.bw - 0.05e9 < state[2] < self.bw + 0.05e9 else 0)
        H = 80 * (state[3])

        reward = (
            A + B + C + D + E + F + G + H
        )

        done = (
            True if reward > 210
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
