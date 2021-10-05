import pandas as pd


class S2P_ANALYSIS(object):
    def __init__(self, expected_snp, s_params, fc1, fc2, passband_return_loss, stopband_insertion_loss):
        with open(expected_snp, 'r') as fp:
            data = fp.readlines()

        starting_index = data.index('!\n')
        for i in range(starting_index + 1):
            data.pop(0)

        formatted_data = []
        for x in range(len(data)):
            tmp = data[x]
            rm_n_tmp = tmp.split('\n')[0]
            rm_t_tmp_list = rm_n_tmp.split('\t')
            formatted_data.append([float(rm_t_tmp_list[0]), float(rm_t_tmp_list[1]), float(rm_t_tmp_list[3])])

        self.s2p_expected_db = pd.DataFrame(formatted_data, columns=['freq', 's11', 's21'])
        self.s2p_actual_db = pd.DataFrame(s_params, columns=['freq', 's11', 's21'])
        self.fc1 = fc1
        self.fc2 = fc2
        self.passband_return_loss = passband_return_loss
        self.stopband_insertion_loss = stopband_insertion_loss

        self.passband_actual = pd.DataFrame()
        self.stopband_actual = pd.DataFrame()

    def test_get_reward(self):
        success_score = 0
        fail_scores = 0

        # Get s11, s21 passband & stopband dB
        self.passband_actual = self.s2p_actual_db.loc[(self.s2p_actual_db['freq'] >= self.fc1) & (self.s2p_actual_db['freq'] <= self.fc2)]
        self.stopband_actual = self.s2p_actual_db.loc[(self.s2p_actual_db['freq'] < self.fc1 - 0.1e9) | (self.s2p_actual_db['freq'] > self.fc2 + 0.1e9)]

        # s21 analysis
        s21_expected_list = self.s2p_expected_db["s21"].values.tolist()
        s21_actual_list = self.s2p_actual_db["s21"].values.tolist()
        for ind in range(len(s21_expected_list)):
            s21_expected = s21_expected_list[ind]
            s21_actual = s21_actual_list[ind]
            if s21_expected - 1 < s21_actual < s21_expected + 1:
                success_score += 100
            else:
                fail_scores += -20

        s11_passband_list = self.passband_actual["s11"].values.tolist()
        s11_stopband_list = self.stopband_actual["s11"].values.tolist()
        if s11_passband_list:
            for s11_db in s11_passband_list:
                if s11_db < self.passband_return_loss:
                    success_score += 100
                else:
                    fail_scores += -20
        if s11_stopband_list:
            for s11_db in s11_stopband_list:
                if 0 > s11_db > -5:
                    success_score += 100
                else:
                    fail_scores += -20

        final_scores = success_score + fail_scores

        if self.s2p_actual_db[self.s2p_actual_db.freq == self.fc1].s21.item() >= -5 and self.s2p_actual_db[self.s2p_actual_db.freq == self.fc2].s21.item() >= -5:
            final_scores += 500
            done = True
        else:
            done = False
        if final_scores < -5500:
            done = True
        if final_scores >= 2000:
            done  = True
        return done, fail_scores

    def get_reward(self):
        done = False

        # Get s11, s21 passband & stopband dB
        self.passband = self.s2p_actual_db.loc[(self.s2p_actual_db['freq'] >= self.fc1) & (self.s2p_actual_db['freq'] <= self.fc2)]
        self.stopband = self.s2p_actual_db.loc[(self.s2p_actual_db['freq'] < self.fc1 - 0.1) | (self.s2p_actual_db['freq'] > self.fc2 + 0.1)]

        # s11 passband (good)
        s11_passband_good = self.passband.loc[self.passband['s11'] < self.passband_return_loss].values.tolist()

        # s11 passband (bad)
        s11_passband_bad = self.passband.loc[self.passband['s11'] > self.passband_return_loss].values.tolist()

        # s11 stopband (good)
        s11_stopband_good = self.stopband.loc[(self.stopband['s11'] > -0.5) & (self.stopband['s11'] <= 0)].values.tolist()

        # s11 stopband (bad)
        s11_stopband_bad = self.stopband.loc[(self.stopband['s11'] < -0.5) | (self.stopband['s11'] > 0)].values.tolist()

        # s21 passband (good)
        s21_passband_good = self.passband.loc[self.passband['s21'] > -3].values.tolist()

        # s21 passband (bad)
        s21_passband_bad = self.passband.loc[self.passband['s21'] < -3].values.tolist()

        # s21 stopband (good)
        s21_stopband_good = self.stopband.loc[self.stopband['s21'] < -20].values.tolist()

        # s21 stopband (bad)
        s21_stopband_bad = self.stopband.loc[self.stopband['s21'] >= -20].values.tolist()

        fail_scores = 0
        success_scores = 0

        # ***************************************** S11 ANALYSIS ******************************************
        # passband
        if s11_passband_bad:
            for x in s11_passband_bad:
                db = x[1]
                diff = abs(self.passband_return_loss - db)
                fail_scores += diff

        if s11_passband_good:
            for x in s11_passband_good:
                db = x[1]
                diff = abs(self.passband_return_loss - db)
                success_scores += diff

        # stopband
        if s11_stopband_bad:
            for x in s11_stopband_bad:
                db = x[1]
                diff = abs(-3 - db)
                fail_scores += diff

        if s11_stopband_good:
            success_scores += (5 * len(s11_stopband_good))

        # ***************************************** S21 ANALYSIS ******************************************
        # In band
        if s21_passband_bad:
            fail_scores += (10 * len(s21_passband_bad))

        if s21_passband_good:
            success_scores += (10 * len(s21_passband_good))

        # stopband
        if s21_stopband_bad:
            for x in s21_stopband_bad:
                db = x[2]
                diff = abs(self.stopband_insertion_loss - db)
                fail_scores += diff

        if s21_stopband_good:
            for x in s21_stopband_good:
                db = x[2]
                if db > -30:
                    success_scores += 1

                elif -30 > db > self.stopband_insertion_loss:
                    success_scores += 3

                elif db <= self.stopband_insertion_loss:
                    success_scores += 5

        final_reward = success_scores - fail_scores

        # Done analysis
        if final_reward > 500 or final_reward < 200:
            done = True
        if self.s2p_actual_db[self.s2p_actual_db.freq == self.fc1].s21.item() >= -5 and self.s2p_actual_db[self.s2p_actual_db.freq == self.fc2].s21.item() >= -5:
            final_reward += 500
            done = True
        if self.s2p_actual_db[self.s2p_actual_db.freq == self.fc1].s21.item() < -5 or self.s2p_actual_db[self.s2p_actual_db.freq == self.fc2].s21.item() < -5:
            final_reward += -1000
            # done = False

        return done, final_reward


# if __name__ == '__main__':
#     s2p_path = r'Test\s2p\ParallelCoupledline_2.s2p'
#     s2p_analysis = S2P_ANALYSIS(s2p_path, 0, 3.4, 3.6, -10, -50)
#     d, r = s2p_analysis.get_reward()
#     print(d, r)
