import pandas as pd


class TestSPara(object):
    def __init__(self, s_actual, s_expected_path):
        with open(s_expected_path, 'r') as fp:
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
        self.s2p_actual_db = pd.DataFrame(s_actual, columns=['freq', 's11', 's21'])

    def export(self):
        export_path = r"Test/s2p/test.xlsx"
        Excelwriter = pd.ExcelWriter(export_path, engine="xlsxwriter")
        df_list = [self.s2p_expected_db, self.s2p_actual_db]
        sheet_list = ["expected", "actual"]
        # We now loop process the list of dataframes
        for i, df in enumerate(df_list):
            df.to_excel(Excelwriter, sheet_name=sheet_list[i], index=False)
        # And finally save the file
        Excelwriter.save()

        # self.s2p_expected_db.to_excel(export_path, sheet_name="s2p_expected", index=False, header=True)
        # self.s2p_actual_db.to_excel(export_path, sheet_name="s2p_actual", index=False, header=True)

