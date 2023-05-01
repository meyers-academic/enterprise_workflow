

import numpy as np
import scipy.sparse as sps
import scipy.linalg as sl
from loguru import logger

from sksparse.cholmod import cholesky

from enterprise.signals import parameter
from enterprise.signals import signal_base
from enterprise.signals import gp_signals


from enterprise_extensions import model_utils, blocks, model_orfs

import enterprise.signals.utils
import enterprise.signals.gp_signals
from enterprise.signals.gp_signals import get_timing_model_basis, BasisGP
from enterprise.signals.parameter import function


@function
def tm_prior(weights, toas, variance=1e-14):
    return weights * variance * len(toas)


def TimingModel(coefficients=False, name="linear_timing_model",
                use_svd=False, normed=True, prior_variance=1e-19):
    """Class factory for marginalized linear timing model signals."""


    basis = get_timing_model_basis(use_svd, normed)
    prior = tm_prior(variance=prior_variance)

    BaseClass = BasisGP(prior, basis, coefficients=coefficients, name=name)

    class TimingModel(BaseClass):
        signal_type = "basis"
        signal_name = "linear timing model"
        signal_id = name + "_svd" if use_svd else name

    return TimingModel

def simulate(pta, params, sparse_cholesky=True):
    logger.info("Simulating residuals...")
    delays, ndiags, fmats, phis = (pta.get_delay(params=params),
                                   pta.get_ndiag(params=params),
                                   pta.get_basis(params=params),
                                   pta.get_phi(params=params))

    residuals = []
    if pta._commonsignals:
        if sparse_cholesky:
            cf = cholesky(sps.csc_matrix(phis))
            gp = np.zeros(phis.shape[0])
            gp[cf.P()] = np.dot(cf.L().toarray(), np.random.randn(phis.shape[0]))
        else:
            gp = np.dot(sl.cholesky(phis, lower=True), np.random.randn(phis.shape[0]))

        i = 0
        for fmat in fmats:
            j = i + fmat.shape[1]
            residuals.append(np.dot(fmat, gp[i:j]))
            i = j

        assert len(gp) == i
    else:
        for fmat, phi in zip(fmats, phis):
            if phi.ndim == 1:
                residuals.append(np.dot(fmat, np.sqrt(phi) * np.random.randn(phi.shape[0])))
            else:
                raise NotImplementedError

    for residual, delay, ndiag in zip(residuals, delays, ndiags):
        if isinstance(ndiag, signal_base.ShermanMorrison):
            logger.info('simulating ecorrs')
            # this code is very slow...
            n = np.diag(ndiag._nvec)
            for j, s in zip(ndiag._jvec, ndiag._slices):
                n[s, s] += j
            residual += np.dot(sl.cholesky(n, lower=True), np.random.randn(n.shape[0]))
        elif ndiag.ndim == 1:
            residual += delay + np.sqrt(ndiag) * np.random.randn(ndiag.shape[0])
        else:
            raise NotImplementedError

    return residuals

def set_residuals(psr, y):
    psr._residuals[psr._isort] = y
