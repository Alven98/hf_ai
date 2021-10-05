import pandas as pd


class S2P_ANALYSIS(object):
    def __init__(self, actual_s, fmin, fmax, fmid, return_loss, insertion_loss):
        self.snp_db = actual_s if str(type(actual_s)) == '<class \'pandas.core.frame.DataFrame\'>' else \
            pd.DataFrame(actual_s, columns=['freq', 's11', 's21'])
        self.snp_db['freq'] = self.snp_db['freq'].div(1e9).round(5)
        # print(self.snp_db)
        self.fmin = fmin / 1e9 if fmin > 1e9 else fmin
        self.fmax = fmax / 1e9 if fmax > 1e9 else fmax
        self.fmid = fmid / 1e9 if fmid > 1e9 else fmid
        self.fc1 = self.fmid - 0.1
        self.fc2 = self.fmid + 0.1
        self.return_loss = return_loss
        self.insertion_loss = insertion_loss

    def get_state_reward(self):
        """
        The get_state_reward for
        1. Construct State
        2. Evaluation

        state (list): The state. Attributes:
              s[0] is the gradient between (fmin, S11dB) to (fc1 - 0.1e9, S11dB)
              s[1] is the probability of S11dB < passband return loss
              s[2] is the gradient between (fc2 + 0.1e9, S11dB) to (fmax, S11dB)
              s[3] is the gradient between (fmin, S21dB) to (fc1 - 0.1e9, S21dB)
              s[4] is the gradient between (fc1, S21dB) to (fc2, S21dB)
              s[5] is the gradient between (fc2 + 0.1e9, S21dB) to (fmax, S21dB)
              s[6] 1 if S21dB of fc1 > -5dB
              s[7] 1 if S21dB of fc2 > -5dB

        returns:
             a: To return the next state, reward and done status.
        """
        A1 = (self.fmin, float(self.snp_db[self.snp_db.freq == self.fmin].s11.item()))
        A2 = (self.fc1 - 0.1, float(self.snp_db[self.snp_db.freq == self.fc1 - 0.1].s11.item()))

        B1 = self.snp_db.loc[(self.fc1 <= self.snp_db.freq) & (self.snp_db.freq <= self.fc2)]
        B2 = B1.loc[B1.s11 < -10]

        C1 = (self.fc2 + 0.1, float(self.snp_db[self.snp_db.freq == self.fc2 + 0.1].s11.item()))
        C2 = (self.fmax, float(self.snp_db[self.snp_db.freq == self.fmax].s11.item()))

        D1 = (self.fmin, float(self.snp_db[self.snp_db.freq == self.fmin].s21.item()))
        D2 = (self.fc1 - 0.1, float(self.snp_db[self.snp_db.freq == self.fc1 - 0.1].s21.item()))

        E1 = (self.fc1, float(self.snp_db[self.snp_db.freq == self.fc1].s21.item()))
        E2 = (self.fc2, float(self.snp_db[self.snp_db.freq == self.fc2].s21.item()))

        F1 = (self.fc2 + 0.1, float(self.snp_db[self.snp_db.freq == self.fc2 + 0.1].s21.item()))
        F2 = (self.fmax, float(self.snp_db[self.snp_db.freq == self.fmax].s21.item()))

        fc1_touch = 1 if self.snp_db[self.snp_db.freq == self.fc1].s21.item() > -5 else 0
        fc2_touch = 1 if self.snp_db[self.snp_db.freq == self.fc2].s21.item() > -5 else 0

        state = [
            (A2[1] - A1[1]) / (A2[0] - A1[0]),
            len(B2) / len(B1),
            (C2[1] - C1[1]) / (C2[0] - C1[0]),
            (D2[1] - D1[1]) / (D2[0] - D1[0]),
            (E2[1] - E1[1]) / (E2[0] - E1[0]),
            (F2[1] - F1[1]) / (F2[0] - F1[0]),
            fc1_touch,
            fc2_touch
        ]

        reward = (
            -100 * (0 if abs(state[0]) < 3 else abs(state[0]))
            - 100 * (1 - state[1])
            - 100 * (0 if abs(state[2]) < 3 else abs(-3 - state[2]))
            - 100 * (0 if 100 < abs(state[3]) < 130 else abs(state[3]))
            - 100 * (0 if abs(state[4]) < 5 else abs(state[4]))
            - 100 * (0 if -130 < state[5] < -100 else abs(state[5]))
            + 100 * (state[1])
            + 100 * fc1_touch
            + 100 * fc2_touch
        )

        done = (
            True if reward > 200
            or (fc1_touch and fc2_touch)
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
