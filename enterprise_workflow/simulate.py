

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
                use_svd=False, normed=True, prior_variance=1e-14):
    """Class factory for marginalized linear timing model signals."""


    basis = get_timing_model_basis(use_svd, normed)
    prior = tm_prior(variance=prior_variance)

    BaseClass = BasisGP(prior, basis, coefficients=coefficients, name=name)

    class TimingModel(BaseClass):
        signal_type = "basis"
        signal_name = "linear timing model"
        signal_id = name + "_svd" if use_svd else name

    return TimingModel


# enterprise.signals.utils.tm_prior = tm_prior
# enterprise.signals.gp_signals.TimingModel = TimingModel
#
# def make_simulation_pta(psrs, noisedict, simulation_params):
#     logger.info("Making PTA")
#     Tspan = model_utils.get_tspan(psrs)
#     tm = gp_signals.TimingModel()
#     wn = blocks.white_noise_block(vary=False, select='backend')
#     rn = blocks.red_noise_block(prior='log-uniform', Tspan=Tspan, components=30)
#
#     log10_A_gw = parameter.Uniform(-18, -12)('log10_A_gw')
#     gamma_gw = parameter.Constant(4.33)('gamma_gw')
#
#     # cpl = enterprise.signals.utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
#     if simulation_params.orf.lower()=="hd":
#         orf = model_orfs.hd_orf()
#     elif simulation_params.orf.lower()=="cp":
#         orf = None
#     if simulation_params.spectral_model=='turnover':
#         logger.info("Simulation has turnover spectrum...")
#         kappa_name = 'kappa_gw'
#         lf0_name = 'log10_fbend_gw'
#         kappa_val = 26/3
#         lf0_gw_val = -8.1
#         kappa_gw = parameter.Constant(kappa_val)(kappa_name)
#         lf0_gw = parameter.Constant(lf0_gw_val)(lf0_name)
#         cpl = enterprise.signals.utils.turnover(log10_A=log10_A_gw, gamma=gamma_gw,
#                              lf0=lf0_gw, kappa=kappa_gw)
#         #crn = blocks.common_red_noise_block(psd='turnover', prior='log-uniform',
#         #                                    orf='hd', Tspan=Tspan, delta_val=8e-9,
#         #                                    components=gwcomponents,
#         #                                    gamma_val=4.33, name='gw')
#         #
#         crn = gp_signals.FourierBasisCommonGP(cpl, model_orfs.hd_orf(),
#                                               components=simulation_params.gw_components,
#                                               combine=True,
#                                               Tspan=Tspan,
#                                               name='gw', pshift=False,
#                                               pseed=None)
#     else:
#         logger.info("Simulation has power_law spectrum")
#         cpl = enterprise.signals.utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
#         crn = gp_signals.FourierBasisCommonGP(cpl, model_orfs.hd_orf(),
#                                               components=simulation_params.gw_components,
#                                               combine=True,
#                                               Tspan=Tspan,
#                                               name='gw', pshift=False,
#                                               pseed=None)
#
#
#     si = tm + wn + rn + crn
#
#     pta = signal_base.PTA([si(psr) for psr in psrs])
#
#     pta.set_default_params(noisedict)
#     return pta

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
