from enterprise.signals.utils import ConditionalGP
import numpy as np

import logging

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

logger = logging.getLogger(__name__)

class Hyperpost(ConditionalGP):
    """
    Similar to `enterprise.signals.utils.ConditionalGP but
    it also includes drawing from the hyperposterior
    for the GP coefficients that is
    not conditional on the data as well. It can do both.
    """
    def _make_hyperpost_no_conditional(self, params):
        phiinvs = self.pta.get_phiinv(params, logdet=False, method=self.phiinv_method)
        if self.pta._commonsignals:

            ch = cholesky(sps.csc_matrix(phiinvs))
            mn = np.zeros(phiinvs.shape[0])

            return ch, mn
        else:
            mns, chs = [], []
            for phiinv in phiinvs:
                phiinv_mat = (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)

                ch = sl.cho_factor(phiinv_mat, lower=True)
                mns.append(np.zeros(phiinv_mat.shape[0]))
                chs.append(np.tril(ch[0]))

            return chs, mns

    def _sample_hyperpost_no_conditional(self, params, n=1, gp=False, variance=True):
        ch, mn = self._make_hyperpost_no_conditional(params)

        ret = []
        b_vals = []
        for j in range(n):
            # since Sigma = L L^T, Sigma^-1 = L^-T L^-1
            # and L^-T x has variance L^-T L^-1 for normal x
            if self.pta._commonsignals:
                b = mn
                if variance:
                    b = b + ch.apply_Pt(ch.solve_Lt(np.random.randn(mn.shape[0]), use_LDLt_decomposition=False))
            else:
                b = np.concatenate(mn)
                if variance:
                    b = b + np.concatenate(
                        [sl.solve_triangular(c.T, np.random.randn(c.shape[0]), lower=False) for c in ch]
                    )
            b_vals.append(b)
            
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
    
    def get_process_from_gp_coeffs(self, coeffs, params, n=1):
        ret = []
        for j in range(len(coeffs)):
#             b = b_vals[j]
            pardict, ntot = {}, 0
            for i, model in enumerate(self.pta.pulsarmodels):
                for sig in model._signals:
                    if sig.signal_type in ["basis", "common basis"]:
                        sb = sig.get_basis(params=params)
#                         nb = sb.shape[1]
#                         print(len(coeffs[j][sig.name + "_coefficients"]))
#                         if nb + ntot > len(coeffs[j][sig.name + "_coefficients"]):
#                             raise IndexError("Missing parameters! You need to set combine=False in your GPs.")

                    
                        pardict[sig.name] = np.dot(sb, coeffs[j][sig.name + "_coefficients"])

#                         ntot += nb

            ret.append(pardict)
        return ret
    
    def get_mean_coefficients_no_conditional(self, params):
        return self._sample_hyperpost_no_conditional(params, n=1, gp=False, variance=False)[0]

    def sample_coefficients_no_conditional(self, params, n=1):
        return self._sample_hyperpost_no_conditional(params, n, gp=False, variance=True)

    def get_mean_processes_no_conditional(self, params):
        return self._sample_hyperpost_no_conditional(params, n=1, gp=True, variance=False)[0]

    def sample_processes_no_conditional(self, params, n=1):
        return self._sample_hyperpost_no_conditional(params, n, gp=True, variance=True)

