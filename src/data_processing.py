import numpy as np


class DataProcessing:

    def __init__(self, df_mat, df_cts):

        self.df_mat = df_mat
        self.df_cts = df_cts

    def get_processed_data(self):

        self.extract_data()
        self.process_data()

        return self.X_eem, self.X_nmr, self.X_lcms

    def extract_data(self):
        data_dic = {}
        dimensions = ["X", "Y", "Z"]
        for dim in dimensions:
            mesurement_technique = self.df_mat[dim]["name"][0][0][0]
            data = self.df_mat[dim]["data"][0][0]
            data_dic[mesurement_technique] = data
            print(f"Extracted {mesurement_technique} data with shape {data.shape}")

        self.X_eem = data_dic["EEM"]
        self.X_nmr = data_dic["3-way NMR"]
        self.X_lcms = data_dic["LCMS"]

    def process_data(self):
        """
        Impute the missing values of X_eem
        """

        X_eem_imputed = self.X_eem.copy()
        means = np.nanmean(self.X_eem, axis=1)
        for i in range(self.X_eem.shape[0]):
            X_eem_imputed[i, :, :] = np.where(
                np.isnan(self.X_eem[i, :, :]), means[i], self.X_eem[i, :, :]
            )

        self.X_eem = X_eem_imputed
