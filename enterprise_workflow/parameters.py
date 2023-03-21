
import pickle
from pathlib import Path
import json
import pandas as pd
import os
import configparser
import numpy as np
from .extra_models import model_registry
from enterprise_workflow import ppc
import time
import re

from loguru import logger
from enterprise_extensions import sampler, models
from enterprise.signals import parameter

from enterprise_extensions.frequentist.optimal_statistic import OptimalStatistic

class ParamsBase(dict):
    """docstring for ParamsBase"""

    def __init__(self, *args, **kwargs):
        super(ParamsBase, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Params(ParamsBase):
    """param object"""

    # allowed sections of params
    allowed_sections = [
        "injection",
        "main_run",
        "reweighting",
        "ostat",
        "ppc"
    ]

    def __init__(self, *args, **kwargs):
        super(Params, self).__init__(*args, **kwargs)

    @classmethod
    def from_config_file(cls, cfile):
        if not os.path.isfile(cfile):
            raise ValueError("config file {0} not found".format(cfile))
        config = configparser.RawConfigParser()
        config.optionxform = str
        config.read(cfile)
        params = cls()
        if "injection" in config:
            params.injection = InjectionParams(**config["injection"])
            print("Running on simulated data.")
        if "main_run" in config:
            params.main_run = BaseRunParams(**config["main_run"])
        # if "reweighting" in config:
        #     params.reweighting = ReweightingParams(**config["reweighting"])
        if "ostat" in config:
            params.ostat = OptStatParams(**config["ostat"])
        if "ppc" in config:
            params.ppc = PPCParams(**config["ppc"])
        return params

    def __str__(self):
        """
        print string for params class.
        """
        msg = "Parameters currently defined\n"
        msg += "-" * len(msg)
        msg += "\n\n\t"
        if "injection" in self:
            msg += 'Injection: ' + self.injection.__str__().replace("\n", "\n\t")
        if "main_run" in self:
            msg += 'Main Run ' + self.main_run.__str__().replace("\n", "\n\t")
        if "reweighting" in self:
            msg += "Reweight: " + self.reweighting.__str__().replace("\n", "\n\t")
        if "ostat" in self:
            msg += 'Opt Stat: ' + self.ostat.__str__().replace("\n", "\n\t")
        if "ppc" in self:
            msg += "ppc: " + self.ppc.__str__().replace("\n", "\n\t")
        return msg

class OptStatParams(ParamsBase):
    valid_params = {
        "pulsar_pickle": (str, None),
        "noise_dictionary": (str, None),
        "chain_feather": (str, None),
        "output_directory": (str, None),
        "pta_model": (str, 'model_2a'),
        "gamma_gw": (float, None),
        'gw_components': (int, 14),
        "human": (str, 'picard'),
        "mcos": (bool, True),
        "scos": (bool, False),
        "num_samples": (int, 1000)
    }

    def __init__(self, *args, **kwargs):
        super(OptStatParams, self).__init__(*args, **kwargs)
        # fill in defaults
        for key, tup in OptStatParams.valid_params.items():
            if key in self:
                continue
            else:
                if tup[0] is bool and isinstance(tup[1], str):
                    if tup[1].lower() == 'true':
                        self[key] = True
                    elif tup[0].lower() == 'false':
                        self[key] = False
                    else:
                        raise ValueError("bool type param must be 'true' or 'false'")
                else:
                    self[key] = tup[1]
        self.check_params()
        self._pta = None

        # used so that we can keep track of whether
        # to use externally defined pta object
        # or one supplied in config
        self._pta_defined_externally = False

    @property
    def pta(self):
        return self._pta

    @pta.setter
    def pta(self, new):
        self._pta = new
        self._pta_defined_externally = True

    def check_params(self):
        for key, val in self.items():
            # pass over search type key
            # check params
            if key in OptStatParams.valid_params:
                if val is not None:
                    if OptStatParams.valid_params[key][0] == bool and isinstance(val, str):
                        if val.lower() == "true":
                            self[key] = True
                        elif val.lower() == "false":
                            self[key] = False
                    else:
                        self[key] = OptStatParams.valid_params[key][0](val)
            else:
                msg = "{0} is not a valid parameter for config type {1}"
                raise ValueError(msg.format(key, "main_run"))

    def run_ostat(self):
        psrs = pickle.load(open(self.pulsar_pickle, 'rb'))
        noisedict = json.load(open(self.noise_dictionary, 'r'))

        for par in list(noisedict.keys()):
            if 'log10_equad' in par:

                efac = re.sub('log10_equad', 'efac', par)
                equad = re.sub('log10_equad', 'log10_t2equad', par)

                noisedict[equad] = np.log10(10 ** noisedict[par] / noisedict[efac])
            elif 'log10_ecorr' in par and 'basis_ecorr' not in par:
                ecorr = par.split('_')[0] + '_basis_ecorr_' + '_'.join(par.split('_')[1:])
                noisedict[ecorr] = noisedict[par]

        # pta = make_simulation_pta(psrs, noisedict, self)
        logger.info(f"Creating pta from {self.pta_model}")
        if self.pta is not None:
            pta = self.pta
        else:
            pta = model_registry[self.pta_model].func(psrs, noisedict=noisedict,
                                                 n_gwbfreqs=self.gw_components,
                                                 gamma_common=self.gamma_gw,
                                                 )
        chain_df = pd.read_feather(self.chain_feather)
        param_names = chain_df.columns
        # chain = np.hstack((chain_df.to_numpy(), np.zeros((len(chain_df), 4))))
        chain = chain_df.to_numpy()
        # print(chain.shape)
        # print(param_names.size)
        results = {}
        if self.scos:
            results['scos'] = {}
            logger.info("HD SCOS")
            os = OptimalStatistic(psrs, pta=pta, bayesephem=False)
            xi, rho, sig, OS_HD, OS_sig_HD = os.compute_noise_marginalized_os(chain, param_names, N=self.num_samples)

            os = OptimalStatistic(psrs, pta=pta, bayesephem=False, orf='monopole')
            xi, rho, sig, OS_MONO, OS_sig_MONO = os.compute_noise_marginalized_os(chain, param_names, N=self.num_samples)

            os = OptimalStatistic(psrs, pta=pta, bayesephem=False, orf='dipole')
            xi, rho, sig, OS_DIPOLE, OS_sig_DIPOLE = os.compute_noise_marginalized_os(chain, param_names, N=self.num_samples)

            results['scos']['rho'] = rho
            results['scos']['xi'] = xi
            results['scos']['sigmas'] = sig
            results['scos']['os_hd'] = OS_HD
            results['scos']['os_hd_sigma'] = OS_HD / OS_sig_HD
            results['scos']['os_mono'] = OS_MONO
            results['scos']['os_mono_sigma'] = OS_MONO / OS_sig_MONO
            results['scos']['os_dipole'] = OS_DIPOLE
            results['scos']['os_dipole_sigma'] = OS_DIPOLE / OS_sig_DIPOLE
        if self.mcos:
            results['mcos'] = {}
            xi, rho, sig, OS, OS_SIG = os.compute_noise_marginalized_multiple_corr_os(chain, param_names=param_names, N=self.num_samples)
            results['mcos']['os_mono'] = OS[:, 0]
            results['mcos']['os_mono_sigma'] = OS_SIG[:, 0]

            results['mcos']['os_dipole'] = OS[:, 1]
            results['mcos']['os_dipole_sigma'] = OS_SIG[:, 1]

            results['mcos']['os_hd'] = OS[:, 2]
            results['mcos']['os_hd_sigma'] = OS_SIG[:, 2]

        outdir = Path(self.output_directory).joinpath("os_results")
        outdir.mkdir(parents=True, exist_ok=True)

        for key in results:
            for key2 in results[key]:
                if isinstance(results[key][key2], np.ndarray):
                    results[key][key2] = results[key][key2].tolist()

        json.dump(results, open(str(outdir.joinpath("os_results.json")), 'w'))



class BaseRunParams(ParamsBase):
    valid_params = {
        "pulsar_pickle": (str, None),
        "noise_dictionary": (str, None),
        "output_directory": (str, None),
        "pta_model": (str, 'model_2a'),
        "gamma_gw": (float, None),
        'gw_components': (int, 14),
        "human": (str, 'picard'),
        "resume": (bool, True),
        "burn_frac": (float, 0.25),
        "num_samples": (int, 1_000_000),
        "tm_marg": (bool, True),
        "tm_svd": (bool, False)
    }

    def __init__(self, *args, **kwargs):
        super(BaseRunParams, self).__init__(*args, **kwargs)
        # fill in defaults
        for key, tup in BaseRunParams.valid_params.items():
            if key in self:
                continue
            else:
                if tup[0] is bool and isinstance(tup[1], str):
                    if tup[1].lower() == 'true':
                        self[key] = True
                    elif tup[0].lower() == 'false':
                        self[key] = False
                    else:
                        raise ValueError("bool type param must be 'true' or 'false'")
                else:
                    self[key] = tup[1]
        self.check_params()
        self._pta = None

        # used so that we can keep track of whether
        # to use externally defined pta object
        # or one supplied in config
        self._pta_defined_externally = False

    @property
    def pta(self):
        return self._pta

    @pta.setter
    def pta(self, new):
        self._pta = new
        self._pta_defined_externally = True

    def check_params(self):
        for key, val in self.items():
            # pass over search type key
            # check params
            if key in BaseRunParams.valid_params:
                if val is not None:
                    if BaseRunParams.valid_params[key][0] == bool and isinstance(val, str):
                        if val.lower() == "true":
                            self[key] = True
                        elif val.lower() == "false":
                            self[key] = False
                    else:
                        self[key] = BaseRunParams.valid_params[key][0](val)
            else:
                msg = "{0} is not a valid parameter for config type {1}"
                raise ValueError(msg.format(key, "main_run"))

    def create_pta_from_registry(self, model_name, *args, **kwargs):
        """
        sets pta attribute for BaseRunParams class

        :param model_name: name of model in registry
        :param args: arguments for model in registry
        :param kwargs: keyword arguments for model in registry

        """
        # should set defined_externally on the fly
        self.pta = model_registry[model_name].func(*args, **kwargs)

    def sample_mcmc(self):
        psrs = pickle.load(open(self.pulsar_pickle, 'rb'))
        noisedict = json.load(open(self.noise_dictionary, 'r'))

        for par in list(noisedict.keys()):
            if 'log10_equad' in par:

                efac = re.sub('log10_equad', 'efac', par)
                equad = re.sub('log10_equad', 'log10_t2equad', par)

                noisedict[equad] = np.log10(10 ** noisedict[par] / noisedict[efac])
            elif 'log10_ecorr' in par and 'basis_ecorr' not in par:
                ecorr = par.split('_')[0] + '_basis_ecorr_' + '_'.join(par.split('_')[1:])
                noisedict[ecorr] = noisedict[par]

        # pta = make_simulation_pta(psrs, noisedict, self)
        logger.info(f"Creating pta from {self.pta_model}")
        if self.pta is not None:
            pta = self.pta
        else:
            pta = model_registry[self.pta_model].func(psrs, noisedict=noisedict,
                                                 n_gwbfreqs=self.gw_components,
                                                 gamma_common=self.gamma_gw,
                                                 tm_marg=self.tm_marg,
                                                 tm_svd=self.tm_svd
                                                 )
        outdir = Path(self.output_directory).joinpath("chain")

        mcmc = sampler.setup_sampler(pta, outdir=str(outdir), resume=self.resume, human=self.human)
        d0 = parameter.sample(pta.params)
        x0 = np.array([d0[par.name] for par in pta.params])

        pta.get_lnlikelihood(x0)
        for ii in range(10):
            pta.get_lnlikelihood(x0)
        start = time.time()
        for ii in range(100):
            pta.get_lnlikelihood(x0)
        end = time.time()
        logger.info(f"Running sampler for {self.num_samples}")
        logger.info(f"Rough estimate of wall time: {np.round((end - start) / 100 * self.num_samples / 3600, 2)} hours")
        mcmc.sample(x0, self.num_samples, SCAMweight=25, AMweight=25, DEweight=50)
        chain = np.loadtxt(str(outdir.joinpath("chain_1.txt")))
        pars = np.loadtxt(str(outdir.joinpath("pars.txt")), dtype=str)
        idx = int(chain.shape[0] * self.burn_frac)
        df = pd.DataFrame({par: chain[idx:, ii] for ii, par in enumerate(pars)})
        df.to_feather(outdir.joinpath("burned_chain.feather"))



class PPCParams(ParamsBase):
    valid_params = {
        "output_directory": (str, "./gp_draw_results"),
        "output_pickle_filename": (str, "gp_results.pkl"),
        "conditional_pta_model": (str, "model_2a"),
        "os_pta_model": (str, "model_2a"),
        "pulsar_pickle": (str, None),
        "noise_dictionary": (str, None),
        "chain_feather": (str, None),
        "gamma_gw": (float, None),
        'gw_components': (int, 14),
        "num_gp_draws": (int, 100)
    }

    def __init__(self, *args, **kwargs):
        super(PPCParams, self).__init__(*args, **kwargs)
        # fill in defaults
        for key, tup in PPCParams.valid_params.items():
            if key in self:
                continue
            else:
                self[key] = tup[1]
        self.check_params()

    def check_params(self):
        for key, val in self.items():
            # pass over search type key
            # check wewave params
            if key in PPCParams.valid_params:
                if val is not None:
                    self[key] = PPCParams.valid_params[key][0](val)
            else:
                msg = "{0} is not a valid parameter for config type {1}"
                raise ValueError(msg.format(key, "ppc"))



    def perform_gp_draws(self):
        logger.info("Loading puolsar and noisedict")
        logger.info(f"pulsar: {self.pulsar_pickle}")
        logger.info(f"noisedict: {self.noise_dictionary}")
        psrs = pickle.load(open(self.pulsar_pickle, 'rb'))
        noisedict = json.load(open(self.noise_dictionary, 'r'))

        for par in list(noisedict.keys()):
            if 'log10_equad' in par:

                efac = re.sub('log10_equad', 'efac', par)
                equad = re.sub('log10_equad', 'log10_t2equad', par)

                noisedict[equad] = np.log10(10 ** noisedict[par] / noisedict[efac])
            elif 'log10_ecorr' in par and 'basis_ecorr' not in par:
                ecorr = par.split('_')[0] + '_basis_ecorr_' + '_'.join(par.split('_')[1:])
                noisedict[ecorr] = noisedict[par]

        # pta = make_simulation_pta(psrs, noisedict, self)
        logger.info(f"Creating pta from {self.os_pta_model} for os")
        print('PTA MODEL OBJECT: ', self.os_pta_model)
        pta_os = model_registry[self.os_pta_model].func(psrs, noisedict=noisedict,
                                                   n_gwbfreqs=self.gw_components,
                                                   )
        logger.info(f"Creating pta from {self.os_pta_model} for conditional")
        pta_cond = model_registry[self.conditional_pta_model].func(psrs, noisedict=noisedict,
                                                              n_gwbfreqs=self.gw_components
                                                              )

        logger.info(f"Loading feather file for chain {self.chain_feather}")
        chain_df = pd.read_feather(self.chain_feather)
        self.gp_samp = ppc.GPSamples(psrs, pta_os, pta_cond, chain_df, num_draws=self.num_gp_draws)
        outfile = Path(self.output_directory).joinpath(self.output_pickle_filename)
        self.gp_samp.sample_all(str(outfile))


class InjectionParams(ParamsBase):
    valid_params = {
        "log10_A_gw": (float, -14),
        "gw_components": (int, 14),
        "gamma_gw": (float, 13 / 3),
        "kappa_gw": (float, None),
        "delta_gw": (float, None),
        "log10_fbend_gw": (float, None),
        # "orf": (str, "HD"),
        # "spectral_model": (str, "power_law"),
        "pta_model": (str, "model_2a"),
        "output_directory": (str, "simulated_data/"),
        "pulsar_pickle": (str, None),
        "noise_dictionary": (str, None),
        "simulation_chain_feather": (str, None),
        "simulation_name": (str, "mysim"),
        "inc_ecorr": (bool, False)
    }

    def __init__(self, *args, **kwargs):
        super(InjectionParams, self).__init__(*args, **kwargs)
        # fill in defaults
        for key, tup in InjectionParams.valid_params.items():
            if key in self:
                continue
            else:
                if tup[0] is bool and isinstance(tup[1], str):
                    if tup[1].lower() == 'true':
                        self[key] = True
                    elif tup[0].lower() == 'false':
                        self[key] = False
                    else:
                        raise ValueError("bool type param must be 'true' or 'false'")
                else:
                    self[key] = tup[1]
        self.check_params()

    def check_params(self):
        for key, val in self.items():
            # pass over search type key
            # check wewave params
            if key in InjectionParams.valid_params:
                if val is not None:
                    self[key] = InjectionParams.valid_params[key][0](val)
            else:
                msg = "{0} is not a valid parameter for config type {1}"
                raise ValueError(msg.format(key, "injection"))

    def simulate(self, params_draw=None):
        from .simulate import simulate, set_residuals
        psrs = pickle.load(open(self.pulsar_pickle, 'rb'))
        noisedict = json.load(open(self.noise_dictionary, 'r'))

        for par in list(noisedict.keys()):
            if 'log10_equad' in par:

                efac = re.sub('log10_equad', 'efac', par)
                equad = re.sub('log10_equad', 'log10_t2equad', par)

                noisedict[equad] = np.log10(10 ** noisedict[par] / noisedict[efac])
            elif 'log10_ecorr' in par and 'basis_ecorr' not in par:
                ecorr = par.split('_')[0] + '_basis_ecorr_' + '_'.join(par.split('_')[1:])
                noisedict[ecorr] = noisedict[par]

        kwargs = {'noisedict': noisedict,
                  'n_gwbfreqs': self.gw_components,
                  'gamma_common': self.gamma_gw}
        if self.pta_model == 'turnover':
            kwargs['kappa'] = self.kappa_gw
            kwargs['lf0_gw'] = self.log10_fbend_gw
        kwargs['simulate'] = True
        pta = model_registry[self.pta_model].func(psrs, **kwargs
                                             )

        if params_draw is None and self.simulation_chain_feather is not None:
            logger.info(f"User specified draw from chain instead of supplying parameters")
            logger.info(f"loading chain from: {self.simulation_chain_feather}")
            chain = pd.read_feather(self.simulation_chain_feather)
            rand = np.random.choice(len(chain))
            params_draw = chain.iloc[rand].to_dict()

        if self.log10_A_gw is not None:
            logger.info("Agw from chain overridden by user.")
            params_draw['gw_log10_A'] = self.log10_A_gw
        if self.gamma_gw is not None:
            logger.info("gw_gamma from chain overridden by user.")
            params_draw['gw_gamma'] = self.gamma_gw

        residuals = simulate(pta, params_draw)
        print(residuals[0][:10])
        logger.info("Setting residuals...")
        for psr, r in zip(psrs, residuals):
            set_residuals(psr, r)
        outdir = Path(self.output_directory).joinpath(f"simulation_{self.simulation_name}")
        outdir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving simulation results in: {outdir}")
        pickle.dump(psrs, open(str(outdir.joinpath("simulated_psrs.pkl")), 'wb'))
        json.dump(params_draw, open(str(outdir.joinpath("simulation_parameters.json")), 'w'))
