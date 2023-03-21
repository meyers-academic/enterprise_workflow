import random
from enterprise_workflow.optimal_statistic import OS
from enterprise.signals import utils
from enterprise_workflow.hyperpost import Hyperpost
import cloudpickle
from tqdm import tqdm
from loguru import logger

from enterprise_extensions.frequentist import optimal_statistic
class SingleGPDraw(object):
    def __init__(self,  os_obj, type='Hypermodel', os_traditional=None):

        self.type = type

        self.snr = os_obj.snr()
        self.os = os_obj.os()
        self.os_sigma = os_obj.os_sigma()

        self.rhos = os_obj.rhos
        self.angles = os_obj.angles
        self.sigmas = os_obj.sigmas

        self.os_freqs = os_obj.os_frequencies()
        self.os_freqs_sigmas = os_obj.os_frequencies_sigmas()

        self.rhos_freqs = os_obj.rhos_freqs
        self.sigmas_freqs = os_obj.sigmas_freqs

        ahat, sigma_tmp = os_obj.mcos()
        self.mcos = ahat
        self.mcos_error = sigma_tmp


class RecoveryObject(object):
    def __init__(self, chain):
        super(RecoveryObject, self).__init__()
        self.j = random.randint(0, len(chain))
        self.params = chain.iloc[self.j]


class GPSamples(object):
    def __init__(self, psrs, main_pta, cpta, chain_df, num_draws=100):
        self.psrs = psrs
        self.pta = main_pta
        self.cpta = cpta
        self.conditional = utils.ConditionalGP(cpta)
        self.hyperposterior = Hyperpost(cpta)
        self.num_draws = num_draws
        self.recovery_list = []
        self.chain_df = chain_df

    def sample_single(self):
        rec = RecoveryObject(self.chain_df)
        os_object = OS(self.psrs, self.pta, rec.params)
        rec.data_os = SingleGPDraw(os_object)

        # conditional
        cond_coeffs = self.conditional.sample_coefficients(rec.params)
        cond_process = self.conditional.get_process_from_gp_coeffs(cond_coeffs, rec.params)
        cres = [sum(cond_process[0][pc.name] for pc in sc if pc.name in cond_process[0]) for sc in self.cpta]
        os_object.set_residuals(cres)

        rec.conditional_os = SingleGPDraw(os_object)
        rec.conditional_os.gp_draws = cond_coeffs
        # gw only
        hres_gw = [sum(cond_process[0][pc.name] for pc in sc if pc.name in cond_process[0] and 'gw' in pc.name) for sc in self.cpta]
        os_object.set_residuals(hres_gw)
        rec.conditional_os_gw_only = SingleGPDraw(os_object)

        # hyperposterior
        hyper_coeffs = self.hyperposterior.sample_coefficients_no_conditional(rec.params)
        hyper_process = self.hyperposterior.get_process_from_gp_coeffs(hyper_coeffs, rec.params)
        # replace timing parameter draws from infinite prior
        # with draws from conditional
        # WARNING: MAY BE BAD!!!
        for key in hyper_process:
            if 'linear_timing' in key:
                hyper_process[0][key] = cond_process[0][key]

        # all
        hres = [sum(hyper_process[0][pc.name] for pc in sc if pc.name in hyper_process[0]) for sc in self.cpta]
        os_object.set_residuals(hres)
        rec.hyperposterior_os = SingleGPDraw(os_object)
        rec.hyperposterior_os.gp_draws = hyper_coeffs
        # gw only
        hres_gw = [sum(hyper_process[0][pc.name] for pc in sc if pc.name in hyper_process[0] and 'gw' in pc.name) for sc in self.cpta]
        os_object.set_residuals(hres_gw)
        rec.hyperposterior_os_gw_only = SingleGPDraw(os_object)
        self.recovery_list.append(rec)

    def sample_all(self, output_file):
        for ii in tqdm(range(self.num_draws)):
            self.sample_single()
            if ii % 10 == 0:
                cloudpickle.dump(self.recovery_list, open(str(output_file), 'wb'))
        cloudpickle.dump(self.recovery_list,  open(str(output_file), 'wb'))


