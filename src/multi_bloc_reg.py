import numpy as np
import tensorly as tl

from tensorly.decomposition import parafac
from tensorly.decomposition import tucker

from tensorly.decomposition._cmtf_als import coupled_matrix_tensor_3d_factorization

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from itertools import product
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


class MultiBlocReg:

    def __init__(self, X_eem, X_nmr, X_lcms, Y):
        self.X_eem = X_eem
        self.X_nmr = X_nmr
        self.X_lcms = X_lcms
        self.X = [X_eem, X_nmr, X_lcms]
        self.Y = Y
        self.SEED = 42

    def CPD(self, ranks, n_iter_max=2000, verbose=True):
        """
        Canonical Polyadic Decomposition (CPD) of the three-way tensor
        (Also known as PARAFAC or CANDECOMP)
        """

        factors_list = []
        for X_str, rank in ranks.items():
            if verbose:
                print(f"{X_str} : {eval('self.'+X_str).shape}) -> (28,{rank})")
            X = eval("self." + X_str)
            if X.shape[0] != 28:
                raise ValueError(
                    "Le mode 0 n'est pas celui des individus. Tenseur mal orienté."
                )
            if X_str in ["X_eem", "X_nmr"]:
                factor = parafac(
                    X,
                    rank=rank,
                    n_iter_max=n_iter_max,
                    tol=1e-6,
                    init="random",
                    random_state=self.SEED,
                )
                factors_list.append(tl.to_numpy(factor[1][0]))

            elif X_str == "X_lcms":
                pca_lcms = PCA(n_components=rank, random_state=self.SEED)
                factor = pca_lcms.fit_transform(X)
                factors_list.append(factor)

        X_all = np.hstack(factors_list)
        return X_all

    def ridge_cross_val(self, X, y, alpha=np.logspace(-3, 3, 100)):

        model = RidgeCV(alphas=alpha, cv=5)
        scores_r2 = cross_val_score(model, X, y, cv=5, scoring="r2")
        scores_rmse = cross_val_score(
            model, X, y, cv=5, scoring="neg_root_mean_squared_error"
        )
        scores_mae = cross_val_score(
            model, X, y, cv=5, scoring="neg_mean_absolute_error"
        )

    def rank_grid_search_rmse_with_plots(self, y, rank_grid):
        """
        Pour chaque combinaison de rangs dans rank_grid, on :
        1) calcule X_all via self.CPD(ranks)
        2) évalue la RMSE moyenne en CV
        3) stocke (ranks, rmse)

        À la fin on trace trois nuages 3D :
        – RMSE vs (rank_eem, rank_nmr)
        – RMSE vs (rank_eem, rank_lcms)
        – RMSE vs (rank_nmr, rank_lcms)
        """
        # 1) Recherche des scores
        results = []
        keys = list(rank_grid.keys())
        all_combos = list(product(*rank_grid.values()))
        for combo in all_combos:
            ranks = dict(zip(keys, combo))
            X_all = self.CPD(ranks, verbose=False)
            # RMSE en CV (valeurs positives)
            neg_rmse = cross_val_score(
                RidgeCV(alphas=np.logspace(-3, 3, 100), cv=5),
                X_all,
                y,
                cv=5,
                scoring="neg_root_mean_squared_error",
            )
            mean_rmse = -np.mean(neg_rmse)
            results.append((ranks, mean_rmse))

        # 2) Passage en tableaux numpy
        rms = np.array([rmse for r, rmse in results])

        # 6) Retour des meilleurs résultats
        best_idx = np.argmin(rms)
        best_combo, best_rmse = results[best_idx]
        return best_combo, best_rmse, results
