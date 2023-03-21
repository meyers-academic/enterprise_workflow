
from enterprise.signals import signal_base
from enterprise.signals import gp_signals, parameter
from enterprise.signals.deterministic_signals import Deterministic
from enterprise.signals import parameter
import enterprise.signals.utils
import enterprise_workflow.simulate as simulate_package

from enterprise.signals.parameter import function

from enterprise_extensions import model_utils, blocks, model_orfs
from loguru import logger
import numpy as np
import re

import enterprise_extensions.models as ee_models
from functools import partial

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


def model2a_sinusoid(psrs, noisedict=None, n_gwbfreqs=14, gamma_common=None, tm_marg=True, simulate=False):
    Tspan = model_utils.get_tspan(psrs)
    if tm_marg:
        tm = gp_signals.MarginalizingTimingModel()
    else:
        tm = gp_signals.TimingModel()
    wn = blocks.white_noise_block(vary=False, select='backend', tnequad=False)
    rn = blocks.red_noise_block(prior='log-uniform', Tspan=Tspan, components=30)

    crn_m2a = blocks.common_red_noise_block(psd='powerlaw', prior='log-uniform',
                                            Tspan=Tspan,
                                            components=n_gwbfreqs,
                                            name='gw', gamma_val=gamma_common)
    dataset_tmin = min([min(psr.toas) for psr in psrs])
    # include sinusoid
    @function
    def sine_wave(toas, flags, A=-9, f=-9, phase=0.0):
        return 10 ** A * np.sin(2 * np.pi * (10 ** f) * (toas - dataset_tmin) + phase)
    def sine_signal(A, f, phase, name=""):
        return Deterministic(sine_wave(A=A, f=f, phase=phase), name=name)
    m1 = sine_signal(A=parameter.Uniform(-9, -4)('common_sin_A'),
                     f=parameter.Uniform(-9, -7)('common_sin_f'),
                     phase=parameter.Uniform(0, 2 * np.pi)('common_sin_phase'))
    # for simulation
    si = tm + wn + rn + crn_m2a + m1
    pta = signal_base.PTA([si(psr) for psr in psrs])

    pta.set_default_params(noisedict)
    return pta

def model_3a_turnover(psrs, noisedict=None, n_gwbfreqs=14, gamma_common=None, tm_marg=True,
                      kappa=26/3, lf0_gw=-8.1, simulate=False):

    Tspan = model_utils.get_tspan(psrs)
    if simulate:
        logger.info("Simulation")
        tm = simulate_package.TimingModel()
    else:
        tm = gp_signals.TimingModel()
    wn = blocks.white_noise_block(vary=False, select='backend')
    rn = blocks.red_noise_block(prior='log-uniform', Tspan=Tspan, components=30)

    log10_A_gw = parameter.Uniform(-18, -12)('gw_log10_A')
    gamma_gw = parameter.Constant(4.33)('gw_gamma')

    # cpl = enterprise.signals.utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
    kappa_name = '{}_kappa'.format('gw')
    lf0_name = '{}_log10_fbend'.format('gw')

    kappa_gw = parameter.Constant(kappa)(kappa_name)
    lf0_gw = parameter.Constant(lf0_gw)(lf0_name)
    cpl = enterprise.signals.utils.turnover(log10_A=log10_A_gw, gamma=gamma_gw,
                         lf0=lf0_gw, kappa=kappa_gw)

    crn = gp_signals.FourierBasisCommonGP(cpl, model_orfs.hd_orf(),
                                          components=n_gwbfreqs, combine=True,
                                          Tspan=Tspan,
                                          name='gw', pshift=False,
                                          pseed=None)

    si = tm + wn + rn + crn

    pta = signal_base.PTA([si(psr) for psr in psrs])

    pta.set_default_params(noisedict)
    return pta

def model3a_sinusoid(psrs, noisedict=None, n_gwbfreqs=14, gamma_common=None, tm_marg=True):
    Tspan = model_utils.get_tspan(psrs)
    if tm_marg:
        tm = gp_signals.MarginalizingTimingModel()
    else:
        tm = gp_signals.TimingModel()

    wn = blocks.white_noise_block(vary=False, select='backend', tnequad=False)
    rn = blocks.red_noise_block(prior='log-uniform', Tspan=Tspan, components=30)

    crn = blocks.common_red_noise_block(psd='powerlaw', prior='log-uniform',
                                        orf='hd', Tspan=Tspan,
                                        components=n_gwbfreqs,
                                        name='gw', gamma_val=gamma_common)
    dataset_tmin = min([min(psr.toas) for psr in psrs])
        # include sinusoid
    @function
    def sine_wave(toas, flags, A=-9, f=-9, phase=0.0):
        return 10 ** A * np.sin(2 * np.pi * (10 ** f) * (toas - dataset_tmin) + phase)
    def sine_signal(A, f, phase, name=""):
        return Deterministic(sine_wave(A=A, f=f, phase=phase), name=name)
    m1 = sine_signal(A=parameter.Uniform(-9, -4)('common_sin_A'),
                     f=parameter.Uniform(-9, -7)('common_sin_f'),
                     phase=parameter.Uniform(0, 2 * np.pi)('common_sin_phase'))
    # for simulation
    si = tm + wn + rn + crn + m1
    pta = signal_base.PTA([si(psr) for psr in psrs])

    pta.set_default_params(noisedict)
    return pta

def model_3d(psrs, noisedict=None, n_gwbfreqs=14, gamma_common=None, tm_marg=False, simulate=False):
    Tspan = model_utils.get_tspan(psrs)
    if simulate:
        tm = simulate_package.TimingModel()
    elif tm_marg:
        tm = gp_signals.MarginalizingTimingModel()
    else:
        tm = gp_signals.TimingModel()

    wn = blocks.white_noise_block(vary=False, select='backend', tnequad=False)
    rn = blocks.red_noise_block(prior='log-uniform', Tspan=Tspan, components=30)

    crn = blocks.common_red_noise_block(psd='powerlaw', prior='log-uniform',
                                        orf='hd', Tspan=Tspan,
                                        components=n_gwbfreqs,
                                        name='gw', gamma_val=gamma_common)
    crn_mono = blocks.common_red_noise_block(psd='powerlaw', prior='log-uniform',
                                        orf='monopole', Tspan=Tspan,
                                        components=n_gwbfreqs,
                                        name='mono', gamma_val=gamma_common)
        # include sinusoid
    # for simulation
    si = tm + wn + rn + crn
    pta = signal_base.PTA([si(psr) for psr in psrs])

    pta.set_default_params(noisedict)
    return pta

def model_2a_or_3a_no_combine(psrs, noisedict=None, n_gwbfreqs=14, gamma_common=None, tm_marg=True, type='model_2a', simulate=False):

    # 15-yr dataset
    Tspan = model_utils.get_tspan(psrs)
    if type == 'model_2a':
        orf = None
    elif type == 'model_3a':
        orf = 'hd'
    else:
        raise ValueError("type must be 'model_2a' or 'model_3a'")

    for par in list(noisedict.keys()):
        if 'log10_equad' in par:
            efac = re.sub('log10_equad', 'efac', par)
            equad = re.sub('log10_equad', 'log10_t2equad', par)

            noisedict[equad] = np.log10(10 ** noisedict[par] / noisedict[efac])
    if simulate:
        tm = simulate_package.TimingModel()
    else:
        tm = gp_signals.TimingModel(use_svd=True)
    wn = ee_models.white_noise_block(vary=False, inc_ecorr=True, tnequad=False, select='backend')
    rn = ee_models.red_noise_block(prior='log-uniform', Tspan=Tspan, components=30, combine=False)
    crn = ee_models.common_red_noise_block(psd='powerlaw', prior='log-uniform', Tspan=Tspan,
                                        components=n_gwbfreqs, gamma_val=gamma_common,
                                        orf=orf, delta_val=None, name='gw',
                                        pshift=False, pseed=None, combine=False)

    si = tm + wn + rn + crn

    pta = signal_base.PTA([si(psr) for psr in psrs])
    pta.set_default_params(noisedict)

    return pta


def model_2a_nocombine(*args, **kwargs):
    kwargs['type'] = 'model_2a'
    return model_2a_or_3a_no_combine(*args, **kwargs)

def model_3a_nocombine(*args, **kwargs):
    kwargs['type'] = 'model_3a'
    return model_2a_or_3a_no_combine(*args, **kwargs)


class ModelClass(object):
    def __init__(self, model_func, model_name, model_use):
        self.func = model_func
        self.name = model_name
        self.use_case = model_use

    def __str__(self):
        out = f'\033[1;31m{self.name}\033[0m: {self.use_case}'
        return out

class ModelRegistry(object):
    def __init__(self, model_list):
        self.model_list = model_list

    def __getitem__(self, item):
        m = [model for model in self.model_list if model.name == item]
        if len(m) == 0:
            raise ValueError(f'Your model, "{item}", does not appear in registry. Here are the available models: \n{self.__str__()}')

        return m[0]

    def __str__(self):
        out = 'Model Registry:\n'
        out += '===============\n'
        for m in self.model_list:
            out += m.__str__()
            out +='\n'
        return out

    def list_models(self):
        print(self)


model_registry = ModelRegistry([ModelClass(ee_models.model_2a, 'model_2a', 'Enterprise extensions model 2a'),
                                ModelClass(ee_models.model_3a, 'model_3a', 'Enterprise extensions model 3a'),
                                ModelClass(model2a_sinusoid, 'm2a_sinusoid', 'Model2a + sinusoid model'),
                                ModelClass(model3a_sinusoid, 'm3a_sinusoid', 'Model3a + sinusoid model'),
                                ModelClass(model_2a_nocombine, 'model_2a_nocombine',
                                           'Model2a, does not combine RN GP coefficients'),
                                ModelClass(model_3a_nocombine, 'model_3a_nocombine',
                                           'Model3a, does not combine RN GP coefficients'),
                                ModelClass(model_3a_turnover, 'turnover', 'M3a turnover'),
                                ModelClass(model_3d, 'model_3d', 'HD plus monopole')])



# model_registry = {'model_2a': ee_models.model_2a,
#                   'model_3a': ee_models.model_3a,
#                   'm2a_sinusoid': model2a_sinusoid,
#                   'm3a_sinusoid': model3a_sinusoid,
#                   'model_2a_nocombine': model_2a_nocombine,
#                   'model_3a_nocombine': model_3a_nocombine
#                   }
