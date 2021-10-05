import pandas as pd


class ReadSNP(object):
    def __init__(self, snp_path):
        with open(snp_path, 'r') as fp:
            data = fp.readlines()

        start_ind = data.index('!\n')
        for i in range(start_ind + 1):
            data.pop(0)

        formatted_data = []
        for x in range(len(data)):
            tmp = data[x]
            rm_n_tmp = tmp.split('\n')[0]
            rm_t_tmp_list = rm_n_tmp.split('\t')
            formatted_data.append([float(rm_t_tmp_list[0]), float(rm_t_tmp_list[1]), float(rm_t_tmp_list[3])])

        self.snp_db = pd.DataFrame(formatted_data, columns=['freq', 's11', 's21'])

    def get_snp_db(self):
        return self.snp_db
