"""
This module contains the implementation of the models.
"""
from itertools import product as c_prod
from functools import reduce
from sympy import symbols
import numpy as np
import pyximport
from direct_lingam.ReLVLiNGAM import ReLVLiNGAM, get_constraints_for_l_latents, MathError
from utils import min_dist_perm, project, cross_moment
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
        return np.polynomial.Polynomial(estimated_coeffs[::-1]).roots().astype(np.float64)

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

        Parameters:
        method (str): The method to use for effect estimation. Can be "ratio" or "cumulant".

        Raises:
        ValueError: If the specified method is not supported.
        """


        if self.highest_l < 2:
            return cross_moment(self.X[:, 0], self.X[:, 1], self.X[:, 2])

        perm = min_dist_perm(self.cumulants_10[1:], self.cumulants_20[1:])
        roots_ratio = [0]
        roots_ratio += [self.roots_20[i + 1]/self.roots_10[perm[i] + 1] for i in range(self.roots_10.shape[0] - 1)]
        match_roots = min_dist_perm(np.array(roots_ratio), self.roots_21)

        return self.roots_21[match_roots[0]]

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
        super().__init__(*args, **kwargs)

    def estimate_effect(self):
        """
        Estimate the causal effect using the specified method.

        Parameters:
        method (str): The method to use for effect estimation. Can be "ratio" or "cumulant".

        Raises:
        ValueError: If the specified method is not supported.
        """
        if self.highest_l < 0:
            perm = min_dist_perm(self.cumulants_10, self.cumulants_20)
            roots_0 = [self.roots_10[0], self.roots_20[perm[0]]]
            roots_1 = [self.roots_10[1], self.roots_20[perm[1]]]
            return self._estimate__effect_cumulant_one_latent(roots_0, roots_1)

        print("estmating effect")
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
        return best_effect

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
        b_2 = ratio_formula(roots_0[0], cov_matrix)
        b_1 = ratio_formula(roots_1[0], cov_matrix)

        diff_1 = np.abs(roots_0[1] - b_1 * roots_0[0])
        diff_2 = np.abs(roots_1[1] - b_2 * roots_1[0])

        if np.argmin([diff_1, diff_2]) == 0:
            return b_1
        return b_2

class IVModel(FixedGraphReLVLiNGAM):
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
        causal_effect = roots[np.argmin(diffs)]
        causal_effect_projected = project(roots[np.argmin(diffs)], self.regs[1:-1], self.regs[-1])
        min_norm = project(np.zeros(2), self.regs[1:-1], self.regs[-1])
        return causal_effect_projected, causal_effect, min_norm
