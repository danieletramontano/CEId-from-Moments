"""
This module contains the implementation of the models.
"""
from itertools import product as c_prod
from functools import reduce
from sympy import symbols
import numpy as np
import networkx as nx
import torch
import pyximport
from direct_lingam.ReLVLiNGAM import ReLVLiNGAM, get_constraints_for_l_latents, MathError
from utils import min_dist_perm, project, cross_moment
from grica.methods import graphical_rica
from scipy import optimize
pyximport.install(inplace=True)


class FixedGraphReLVLiNGAM(ReLVLiNGAM):
    """
    FixedGraphReLVLiNGAM class.

    Parameters:
    *args: positional arguments

    **kwargs: keyword arguments
    """
    def __init__(self,
                 *args,
                 **kwargs):
        """
        Initialize the FixedGraphReLVLiNGAM model.

        Parameters:
        *args: positional arguments
        **kwargs: keyword arguments
        """
        super().__init__(*args, **kwargs)
        self.cumulants = self._estimate_cumulants(self.X)

    def _get_roots(self, i=1, j=0, highest_l=None, first_zero=False, cumulants = None):
        """
        Get the roots of the polynomial equations for the specified nodes.

        Parameters:
        i (int): The index of the first node.
        j (int): The index of the second node.
        highest_l (int): The highest order of latent variables.

        Returns:
        numpy.ndarray: The roots of the polynomial equations.
        """

        if highest_l is None:
            highest_l = self.highest_l
        if cumulants is None:
            cumulants = self.cumulants

        equations_bij = get_constraints_for_l_latents(highest_l)["equations_bij"]

        eq = equations_bij[0]
        specify_nodes = {
            sym: symbols(
                sym.name[:2] + "".join(sorted(sym.name[2:].replace("j", str(j)).replace("i", str(i))))
            )
            for sym in reduce(set.union, [eq.free_symbols for eq in equations_bij])
            if str(sym) != "b_ij"
        }
        symbols_to_cumulants = self._form_symbol_to_cumulant_dict(cumulants, [i, j], scale_partly=False)
        estimated_coeffs = [float(coeff.subs(specify_nodes).subs(symbols_to_cumulants)) for coeff in eq.all_coeffs()]
        if first_zero:
            estimated_coeffs[-1] = 0
        return np.polynomial.Polynomial(estimated_coeffs[::-1]).roots().real.astype(np.float64)

    def _marginal__cumulants(self, candidate_effects, j=0, i=1):
        """
        Estimate the marginal cumulants for the specified nodes.

        Parameters:
        j (int): The index of the first node.
        i (int): The index of the second node.

        Returns:
        numpy.ndarray: The estimated marginal cumulants.
        """
        source, other_node = j, i
        l = self.highest_l  # The number of latent variables confounding 1 and 2
        # For k < l+1, the marginal omega cannot be inferred.
        k = l + 1
        b_tilde = [candidate_effects ** i for i in range(k)]
        symbols_to_cumulants = self._form_symbol_to_cumulant_dict(self.cumulants, [i, j], scale_partly=False)
        y = np.array(
            [
                float(symbols_to_cumulants[symbols(f"c_{''.join(sorted((str(source),) * (k - i) + (str(other_node),) * i))}")])
                for i in range(k)
            ]
        )
        try:
            marginal_omegas = np.linalg.lstsq(b_tilde, y, rcond=None)[0]
        except np.linalg.LinAlgError as exc:
            raise MathError(f"Linear system for {k}th order omega for source {source} and test node {other_node} is singular.") from exc
        return marginal_omegas

    def _get_grica_estimate(self, graph, latent = None):
        """
        Estimate the causal effect using the specified method.
        """

        lr = 0.1
        w_init = 'cov_guess'
        momentum = 0
        epochs = 400
        lambda_grica = 0
        torch_data = torch.tensor(self.X, dtype=torch.float32)
        d_cov = (torch_data.t()).cov()

        # Whitening
        _, s, v = d_cov.svd()
        s_2=torch.inverse(torch.diag(s.sqrt()))
        w_w = s_2.matmul(v.t())
        data_whitened = w_w.matmul(torch_data.t()).t()
        if latent is None:
            latent = self.highest_l
        observed = self.X.shape[1]
        weight_pred= graphical_rica(latent,
                                    observed,
                                    graph,
                                    torch_data,
                                    data_whitened,
                                    epochs,
                                    lr,
                                    w_w,
                                    w_init,
                                    momentum,
                                    lambda_grica)

        return weight_pred

class CM(FixedGraphReLVLiNGAM):
    """
    CM class.

    Parameters:
    *args: positional arguments

    **kwargs: keyword arguments

    """
    def __init__(self,
                 *args,
                 **kwargs):
        """
            Initialize the DiDLiNGAM model.

            Parameters:
            *args: positional arguments
            **kwargs: keyword arguments
        """
        super().__init__(*args, **kwargs)
        if self.highest_l < 2:
            kwargs["first_zero"] = False
        first_zero = kwargs.get("first_zero", True)

        self.roots_10 = np.array(sorted(self._get_roots(i=1, j=0, first_zero=first_zero), key = np.abs))
        self.roots_20 = np.array(sorted(self._get_roots(i=2, j=0, first_zero=first_zero), key = np.abs))
        self.roots_21 = np.array(sorted(self._get_roots(i=2, j=1), key = np.abs))
        self.cumulants_10 = self._marginal__cumulants(self.roots_10, j=0, i=1)
        self.cumulants_20 = self._marginal__cumulants(self.roots_20, j=0, i=2)

    def estimate_effect(self):
        """
        Estimate the causal effect using the specified method.
        """
        perm = min_dist_perm(self.cumulants_10[1:], self.cumulants_20[1:])
        roots_ratio = [0]
        roots_ratio += [self.roots_20[i + 1]/self.roots_10[perm[i] + 1] for i in range(self.roots_10.shape[0] - 1)]
        match_roots = min_dist_perm(np.array(roots_ratio), self.roots_21)

        return self.roots_21[match_roots[0]]

    def estimate_effect_cross_moment(self):
        """
        Estimate the causal effect using the specified method.
        """
        return cross_moment(self.X[:, 0], self.X[:, 1], self.X[:, 2])

    def estimate_effect_grica(self):
        """
        Estimate the causal effect using the specified method.
        """
        latent_array = [0] * self.highest_l + [1] * 3
        empty_array = [0] * (self.highest_l + 3)
        treatment_array = [0] * (self.highest_l +2) + [1]

        graph_adjacency = [latent_array]*self.highest_l + [empty_array] + [treatment_array] + [empty_array]
        graph = nx.DiGraph(np.array(graph_adjacency))

        weight_pred = self._get_grica_estimate(graph)
        return weight_pred[-1].item()


class ICM(CM):
    """
    ICM class.

    Parameters:
    *args: positional arguments

    **kwargs: keyword arguments

    """
    def __init__(self,
                 *args,
                 **kwargs):
        """
            Initialize the DiDLiNGAM model.

            Parameters:
            *args: positional arguments
            **kwargs: keyword arguments
        """

        kwargs["first_zero"] = False
        self.cumulant_estimate = None
        super().__init__(*args, **kwargs)

    def estimate_effect(self):
        """
        Estimate the causal effect using the specified method.

        Parameters:
        method (str): The method to use for effect estimation. Can be "ratio" or "cumulant".

        Raises:
        ValueError: If the specified method is not supported.
        """
        if self.highest_l < 2:
            perm = min_dist_perm(self.cumulants_10, self.cumulants_20)
            roots_0 = [self.roots_10[0], self.roots_20[perm[0]]]
            roots_1 = [self.roots_10[1], self.roots_20[perm[1]]]
            return self._estimate__effect_cumulant_one_latent(roots_0, roots_1)

        perm = min_dist_perm(self.cumulants_10, self.cumulants_20)
        min_off = np.inf
        best_effect = None
        for i in zip(self.roots_10, self.roots_20[perm]):
            z, t, y = self.X[:,0], self.X[:,1], self.X[:,2]
            t = t - i[0]*z
            y = y - i[1]*z
            cm_sample = np.vstack((z, t, y)).T
            cm_model = CM(cm_sample, highest_l=self.highest_l)
            cm_effect = cm_model.estimate_effect()
            offset = np.abs(cm_effect * i[1] - i[0])
            if offset < min_off:
                min_off = offset
                best_effect = cm_effect

        self.cumulant_estimate = best_effect
        return self.cumulant_estimate

    def _estimate__effect_cumulant_one_latent(self, roots_0, roots_1):
        """
        Estimate the causal effect using the cumulant method.

        Parameters:
        match (list): The matching indices for the marginal cumulants.

        Returns:
        float: The estimated causal effect.
        """
        def ratio_formula(root, cov_matrix):
            return (cov_matrix[1, 2] - root * cov_matrix[0, 2]) / (cov_matrix[1, 1] - root * cov_matrix[0, 1])

        cov_matrix = self.cumulants.get(2)
        # Estimate the causal effect
        b_1 = ratio_formula(roots_0[0], cov_matrix)
        b_0 = ratio_formula(roots_1[0], cov_matrix)

        diffs = [np.abs(roots_0[1] - b_0 * roots_0[0]), np.abs(roots_1[1] - b_1 * roots_1[0])]

        if np.argmin(diffs) == 0:
            self.cumulant_estimate = b_0
            return self.cumulant_estimate
        self.cumulant_estimate = b_1
        return self.cumulant_estimate

    def estimate__effect_minimization(self):
        Z, T, Y = self.X[:, 0], self.X[:, 1], self.X[:, 2]
        if self.cumulant_estimate is None:
            self.estimate_effect()

        init_point =  self.cumulant_estimate
        alpha = 1

        YYZ = (Y*Y*Z).mean()
        YTZ = (Y*T*Z).mean()
        TTZ = (T*T*Z).mean()
        YTT = (Y*T*T).mean()
        TTT = (T*T*T).mean()

        a0 = YYZ * YTT
        a1 = -TTT*YYZ - 2*YTZ*YTT
        a2 = TTZ*YTT + 2*YTZ*TTT
        a3 = -TTZ*TTT

        YZZ = (Y*Z*Z).mean()
        TZZ = (T*Z*Z).mean()
        YYT = (Y*Y*T).mean()
        YTT = (Y*T*T).mean()
        TTT = (T*T*T).mean()

        b0 = YZZ*YYT
        b1 = -TZZ*YYT - 2*YTT*YZZ
        b2 = YZZ*TTT + 2*YTT*TZZ
        b3 = -TZZ*TTT

        cov21 = (Y*T).mean()
        cov20 = (Y*Z).mean()
        cov11 = (T*T).mean()
        cov10 = (T*Z).mean()

        def f_min_reg(x):
            k = (a0 + a1*x + a2*x*x + a3*x*x*x) / (b0 + b1*x + b2*x*x + b3*x*x*x)

            return (x - (cov21 - k*cov20) / (cov11-k*cov10))**2 + alpha*(x - init_point)**2

        sol = optimize.minimize(f_min_reg, init_point, method='BFGS')
        return sol.x[0]

    def estimate_effect_grica(self):
        """
        Estimate the causal effect using the specified method.
        """
        latent_array = [0] * self.highest_l + [1] * 3
        proxy_array = [0] * (self.highest_l + 1) + [1, 0]
        treatment_array = [0] * (self.highest_l +2) + [1]
        effect_array = [0] * (self.highest_l + 3)
        graph_adjacency = [latent_array]*self.highest_l + [proxy_array] + [treatment_array] + [effect_array]
        graph = nx.DiGraph(np.array(graph_adjacency))

        weight_pred = self._get_grica_estimate(graph)
        return weight_pred[-1].item()

class IV(FixedGraphReLVLiNGAM):
    """
    IVModel class.

    Parameters:
    *args: positional arguments

    **kwargs: keyword arguments

    """
    def __init__(self,
                 *args,
                 **kwargs):
        """
            Initialize the DiDLiNGAM model.

            Parameters:
            *args: positional arguments
            **kwargs: keyword arguments
        """
        super().__init__(*args, **kwargs)
        cov_matrix = self.cumulants.get(2)
        self.regs = cov_matrix[0, :]/cov_matrix[0, 0]

        x_reg = self.X[:, [0, 1, 2, 3]]
        x_reg -= np.concatenate([self.X[:, [0]]*r for r in self.regs], axis=1)

        self.red_cumulants = self._estimate_cumulants(np.asfortranarray(x_reg))
        self.roots_1 = self._get_roots(i=3, j=1, cumulants = self.red_cumulants)
        self.roots_2 = self._get_roots(i=3, j=2, cumulants = self.red_cumulants)

    def estimate_effect(self):
        """
        Estimate the causal effect using the specified method.

        Parameters:
        method (str): The method to use for effect estimation. Can be "ratio" or "cumulant".

        Raises:
        ValueError: If the specified method is not supported.
        """
        roots = list(c_prod(self.roots_1, self.roots_2))
        diffs = [np.abs(self.regs[1]*r[0] + self.regs[2]*r[1] - self.regs[3]) for r in roots]
        causal_effect_projected = project(roots[np.argmin(diffs)], self.regs[1:-1], self.regs[-1])
        return causal_effect_projected

    def estimate_effect_min_norm(self):
        """
        Estimate the causal effect using the specified method.
        """
        min_norm = project(np.zeros(2), self.regs[1:-1], self.regs[-1])
        return min_norm

    def estimate_effect_grica(self):
        """
        Estimate the causal effect using the specified method.
        """
        latent_array_1 = [0] * (2*self.highest_l+ 1) + [1, 0, 1]
        latent_array_2 = [0] * (2*self.highest_l + 1) + [0, 1, 1]
        instrument_array = [0] * (2*self.highest_l + 1) + [1, 1, 0]
        treatment_array = [0] * (2*self.highest_l + 1) + [0, 0, 1]
        effect_array = [0] * (2*self.highest_l + 4)

        graph_adjacency = [latent_array_1]*self.highest_l + [latent_array_2]*self.highest_l
        graph_adjacency += [instrument_array]
        graph_adjacency += [treatment_array]*2
        graph_adjacency +=[effect_array]

        graph = nx.DiGraph(np.array(graph_adjacency))

        weight_pred = self._get_grica_estimate(graph = graph, latent = 2*self.highest_l)
        return np.array(weight_pred[-2:])


class IVModelNEW(FixedGraphReLVLiNGAM):
    """
    IVModel class.

    Parameters:
    *args: positional arguments

    **kwargs: keyword arguments

    """
    def __init__(self,
                 *args,
                 instruments = None,
                 treatments = None,
                 outcome = None,
                 **kwargs):
        """
            Initialize the DiDLiNGAM model.

            Parameters:
            *args: positional arguments
            **kwargs: keyword arguments
        """
        super().__init__(*args, **kwargs)
        cov_matrix = self.cumulants.get(2)
        self.regs = np.invert(cov_matrix[instruments, instruments])*cov_matrix[instruments, :]

        x_reg = self.X[:, [0, 1, 2, 3]] ##TO BE CHANGED
        x_reg -= np.concatenate([self.X[:, instruments]*r for r in self.regs], axis=1)

        self.red_cumulants = self._estimate_cumulants(np.asfortranarray(x_reg))
        self.roots = [self._get_roots(i=outcome[0], j=treatment, cumulants = self.red_cumulants) for treatment in treatments]

    def estimate_effect(self):
        """
        Estimate the causal effect using the specified method.

        Parameters:
        method (str): The method to use for effect estimation. Can be "ratio" or "cumulant".

        Raises:
        ValueError: If the specified method is not supported.
        """
        roots = list(c_prod(self.roots))
        diffs = [np.abs(self.regs[1]*r[0] + self.regs[2]*r[1] - self.regs[3]) for r in roots]
        causal_effect_projected = project(roots[np.argmin(diffs)], self.regs[1:-1], self.regs[-1])
        min_norm = project(np.zeros(2), self.regs[1:-1], self.regs[-1])
        return causal_effect_projected, min_norm

    def estimate_effect_grica(self):
        """
        Estimate the causal effect using the specified method.
        """
        latent_array_1 = [0] * (2*self.highest_l+ 1) + [1, 0, 1]
        latent_array_2 = [0] * (2*self.highest_l + 1) + [0, 1, 1]
        instrument_array = [0] * (2*self.highest_l + 1) + [1, 1, 0]
        treatment_array = [0] * (2*self.highest_l + 1) + [0, 0, 1]
        effect_array = [0] * (2*self.highest_l + 4)

        graph_adjacency = np.array([latent_array_1]*self.highest_l + [latent_array_2]*self.highest_l + [instrument_array] + [treatment_array]*2 + [effect_array])
        graph = nx.DiGraph(graph_adjacency)

        weight_pred = self._get_grica_estimate(graph = graph, latent = 2*self.highest_l)
        return weight_pred[-2:]
