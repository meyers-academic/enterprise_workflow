import numpy as np
import scipy.linalg as sl
import scipy.sparse as sps
import scipy.special as ss
from pkg_resources import Requirement, resource_filename
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from sksparse.cholmod import cholesky

import enterprise
from enterprise import constants as const
from enterprise import signals as sigs  # noqa: F401
from enterprise.signals.gp_bases import (  # noqa: F401
    createfourierdesignmatrix_dm,
    createfourierdesignmatrix_env,
    createfourierdesignmatrix_eph,
    createfourierdesignmatrix_ephem,
    createfourierdesignmatrix_red,
)
from enterprise.signals.gp_priors import powerlaw, turnover  # noqa: F401
from enterprise.signals.parameter import function



class ConditionalGP:
    def __init__(self, pta, phiinv_method="cliques"):
        """This class allows the computation of conditional means and
        random draws for all GP coefficients/realizations in a model,
        given a vector of hyperparameters. It currently requires combine=False
        for all GPs (or otherwise distinct bases) and does not
        work with MarginalizingTimingModel (fast new-style likelihood)."""

        self.pta = pta
        self.phiinv_method = phiinv_method

    def _make_conditional(self, params):
        TNrs = self.pta.get_TNr(params)
        TNTs = self.pta.get_TNT(params)
        phiinvs = self.pta.get_phiinv(params, logdet=False, method=self.phiinv_method)

        # TO DO: all this could be more efficient
        #        also it's unclear if it works with MarginalizingTimingModel
        if self.pta._commonsignals:
            TNr = np.concatenate(TNrs)
            Sigma = sps.block_diag(TNTs, "csc") + sps.csc_matrix(phiinvs)

            ch = cholesky(Sigma)
            mn = ch(TNr)

            return ch, mn
        else:
            mns, chs = [], []
            for TNr, TNT, phiinv in zip(TNrs, TNTs, phiinvs):
                Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)

                ch = sl.cho_factor(Sigma, lower=True)
                mns.append(sl.cho_solve(ch, TNr))
                chs.append(np.tril(ch[0]))

            return chs, mns

    def _sample_conditional(self, params, n=1, gp=False, variance=True, mean=True):
        ch, mn = self._make_conditional(params)

        ret = []
        for j in range(n):
            # since Sigma = L L^T, Sigma^-1 = L^-T L^-1
            # and L^-T x has variance L^-T L^-1 for normal x
            if self.pta._commonsignals:
                if mean:
                    b = mn
                else:
                    b = np.zeros(mn.size)
                if variance:
                    b = b + ch.apply_Pt(ch.solve_Lt(np.random.randn(mn.shape[0]), use_LDLt_decomposition=False))


            else:
                b = np.concatenate(mn)
                if not mean:
                    b = np.zeros(b.size)
                if variance:
                    b = b + np.concatenate(
                        [sl.solve_triangular(c.T, np.random.randn(c.shape[0]), lower=False) for c in ch]
                    )

            pardict, ntot = {}, 0
            for i, model in enumerate(self.pta.pulsarmodels):
                for sig in model._signals:
                    if sig.signal_type in ["basis", "common basis"]:
                        sb = sig.get_basis(params=params)
                        nb = sb.shape[1]

                        if nb + ntot > len(b):
                            raise IndexError("Missing parameters! You need to set combine=False in your GPs.")

                        if gp:
                            pardict[sig.name] = np.dot(sb, b[ntot : nb + ntot])
                        else:
                            pardict[sig.name + "_coefficients"] = b[ntot : nb + ntot]

                        ntot += nb

            ret.append(pardict)

        return ret

    def get_process_from_gp_coeffs(self, coeffs, params, n=1, gw_pshifts=False):
        ret = []
        for j in range(len(coeffs)):
#             b = b_vals[j]
            pardict, ntot = {}, 0
            for i, model in enumerate(self.pta.pulsarmodels):
                for sig in model._signals:
                    if sig.signal_type in ["basis", "common basis"]:
                        sb = sig.get_basis(params=params)
                        if gw_pshifts and 'gw' in sig.name:
                            gw_coeff_size = int(coeffs[j][sig.name + "_coefficients"].size / 2)
                            randphases = np.random.rand(gw_coeff_size) * 2 * np.pi
                            gw_coeffs = coeffs[j][sig.name + "_coefficients"]
                            cmplx = (gw_coeffs[::2] + 1j * gw_coeffs[1::2]) * np.exp(1j * randphases)
                            gw_coeffs[::2] = np.real(cmplx)
                            gw_coeffs[1::2] = np.imag(cmplx)
                            pardict[sig.name] = np.dot(sb, gw_coeffs)
                        else:
                            pardict[sig.name] = np.dot(sb, coeffs[j][sig.name + "_coefficients"])


            ret.append(pardict)
        return ret

    def get_mean_coefficients(self, params):
        return self._sample_conditional(params, n=1, gp=False, variance=False)[0]

    def sample_coefficients(self, params, n=1, mean=True):
        return self._sample_conditional(params, n, gp=False, variance=True, mean=mean)

    def get_mean_processes(self, params):
        return self._sample_conditional(params, n=1, gp=True, variance=False)[0]

    def sample_processes(self, params, n=1, mean=True):
        return self._sample_conditional(params, n, gp=True, variance=True, mean=mean)


