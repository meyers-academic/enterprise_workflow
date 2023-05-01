from sksparse.cholmod import cholesky
import scipy.sparse as sps
import numpy as np
import scipy.linalg
from enterprise.signals import utils
import math
from tqdm import tqdm

def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"${0} \times 10^{{{1}}}$".format(base, int(exponent))
    else:
        return float_str

def partition(a, n):
    k = len(a) // n
    ps = [a[i:i+k] for i in range(0,len(a),k)]

    return ps

def partition2(a, n):
    js = np.linspace(0, len(a) + 1, n + 1).astype(int)
    return [a[js[i]:js[i + 1]] for i in range(n)]


def weightedavg(rho, sig):
    avg = sum(r / s**2 for r, s in zip(rho, sig))
    weights = sum(1 / s**2 for s in sig)

    return (avg / weights, math.sqrt(1 / weights))


def makebins(os, bins, angles=None):
    angles = os.angles if angles is None else angles

    return np.array([(np.mean(angles[p]), *weightedavg(os.rhos[p], os.sigmas[p]))
                    for p in partition2(np.argsort(angles), bins)])

class Cmat:
    def __init__(self, sm, params):
        self.Nmat = sm.get_ndiag(params)
        self.Sigma = scipy.linalg.cho_factor(sm.get_TNT(params) + np.diag(sm.get_phiinv(params)))
        self.Fmat = sm.get_basis(params)

    def solve(self, r, l):
        FNr = self.Nmat.solve(r, self.Fmat)
        FNl = FNr if (l is r) else self.Nmat.solve(l, self.Fmat)

        return self.Nmat.solve(l, r) - np.dot(FNl.T, scipy.linalg.cho_solve(self.Sigma, FNr)).T

class OS:
    def __init__(self, psrs, pta, params, residuals=None):
        self.psrs, self.params = psrs, params
        self.pairs  = [(i,j) for i in range(len(pta)) for j in range(i+1, len(pta))]
        self.angles = np.array([np.arccos(np.dot(self.psrs[i].pos, self.psrs[j].pos)) for (i,j) in self.pairs])

        ys = [sm._residuals for sm in pta] if residuals is None else residuals
        self.Cs = [Cmat(sm, params) for sm in pta]

        self.Fgws  = [sm['gw'].get_basis(params) for sm in pta]
        self.Gamma = np.array(pta[0]['gw'].get_phi({**params, 'gw_log10_A': 0})).copy()

        self.phimat = pta[0]['gw'].get_phi({**params, 'gw_log10_A': 0, 'gw_gamma': 0})

        Nfreqs = int(self.Gamma.size / 2)

        FCys = [C.solve(y, Fgw) for C, y, Fgw in zip(self.Cs, ys, self.Fgws)]

        self.FCFs = [C.solve(Fgw, Fgw) for C, Fgw in zip(self.Cs, self.Fgws)]

        # a . np.diag(g) . b = a . (g * b) = (a * g) . b
        ts = [np.dot(FCys[i], self.Gamma * FCys[j]) for (i,j) in self.pairs]
        # A . np.diag(g) . B = (A * g) . B
        self.bs = [np.trace(np.dot(self.FCFs[i] * self.Gamma, self.FCFs[j] * self.Gamma)) for (i,j) in self.pairs]

        self.rhos = np.array(ts) / np.array(self.bs)
        self.sigmas = 1.0 / np.sqrt(self.bs)

        self.rhos_freqs = np.zeros((Nfreqs, self.rhos.size))
        self.sigmas_freqs = np.zeros((Nfreqs, self.rhos.size))

        for ii in range(Nfreqs):
            # pick out just want we want
            gamma_tmp = np.zeros(Nfreqs * 2)
            gamma_tmp[2*ii:2*(ii+1)] = self.phimat[2*ii:2*(ii+1)]
            # a . np.diag(g) . b = a . (g * b) = (a * g) . b
            ts_tmp = [np.dot(FCys[i], gamma_tmp * FCys[j]) for (i,j) in self.pairs]
            # A . np.diag(g) . B = (A * g) . B
            bs_tmp = [np.trace(np.dot(self.FCFs[i] * gamma_tmp, self.FCFs[j] * gamma_tmp)) for (i,j) in self.pairs]

            self.rhos_freqs[ii] = np.array(ts_tmp) / np.array(bs_tmp)
            self.sigmas_freqs[ii] = 1.0 / np.sqrt(bs_tmp)


    def set_residuals(self, residuals):
        ys = residuals

        FCys = [C.solve(y, Fgw) for C, y, Fgw in zip(self.Cs, ys, self.Fgws)]
        ts = [np.dot(FCys[i], self.Gamma * FCys[j]) for (i,j) in self.pairs]
        Nfreqs = int(self.Gamma.size / 2)
        self.rhos = np.array(ts) / np.array(self.bs)

        self.rhos_freqs = np.zeros((Nfreqs, self.rhos.size))
        self.sigmas_freqs = np.zeros((Nfreqs, self.rhos.size))

        for ii in range(Nfreqs):
            # pick out just want we want
            gamma_tmp = np.zeros(Nfreqs * 2)
            # gamma_tmp[2*ii:2*(ii+1)] = self.Gamma[2*ii:2*(ii+1)]
            gamma_tmp[2*ii:2*(ii+1)] = self.phimat[2*ii:2*(ii+1)]
            # a . np.diag(g) . b = a . (g * b) = (a * g) . b
            ts_tmp = [np.dot(FCys[i], gamma_tmp * FCys[j]) for (i,j) in self.pairs]
            # A . np.diag(g) . B = (A * g) . B
            bs_tmp = [np.trace(np.dot(self.FCFs[i] * gamma_tmp, self.FCFs[j] * gamma_tmp)) for (i,j) in self.pairs]

            self.rhos_freqs[ii] = np.array(ts_tmp) / np.array(bs_tmp)
            self.sigmas_freqs[ii] = 1.0 / np.sqrt(bs_tmp)

    def setup_mcos(self, orfs):
        if not hasattr(self, "_mcos_ready") or not self._mcos_ready:
            self.design = np.array([[orfs[kk](self.psrs[i].pos, self.psrs[j].pos) for (i,j) in self.pairs]
                           for kk in range(len(orfs))]).T
            self.sigma_inv_matrix = np.linalg.inv(self.design.T @ np.linalg.inv(np.diag(self.sigmas**2)) @ self.design)
            self.prefac_matrix = self.sigma_inv_matrix @ self.design.T @ np.diag(self.sigmas**-2)
            self._mcos_ready = True


    def mcos(self, orfs=[utils.monopole_orf, utils.dipole_orf, utils.hd_orf]):
        self.setup_mcos(orfs)
        ahat = self.prefac_matrix @ self.rhos
        return ahat, np.sqrt(np.diag(self.sigma_inv_matrix))



    def _set_orf(self, orf):
        if not hasattr(self, '_orf') or self._orf is not orf:
            self.orfs = np.array([orf(self.psrs[i].pos, self.psrs[j].pos) for (i,j) in self.pairs])
            self._orf = orf

    def os(self, orf=utils.hd_orf, sel=(lambda p: slice(None))):
        self._set_orf(orf)

        mask = sel(self.pairs)
        return (np.sum(self.rhos[mask] * self.orfs[mask] / self.sigmas[mask]**2) /
                np.sum(self.orfs[mask]**2 / self.sigmas[mask]**2))

    def os_frequencies(self, orf=utils.hd_orf, sel=(lambda p: slice(None))):
        self._set_orf(orf)

        mask = sel(self.pairs)

        return np.sum(self.orfs * self.rhos_freqs * self.sigmas_freqs**-2, axis=1) / np.sum(self.orfs**2 * self.sigmas_freqs**-2, axis=1)

    def os_frequencies_sigmas(self, orf=utils.hd_orf, sel=(lambda p: slice(None))):
        self._set_orf(orf)

        mask = sel(self.pairs)

        return np.sum(self.orfs**2 * self.sigmas_freqs**-2, axis=1)**-0.5



    def os_sigma(self, orf=utils.hd_orf, sel=(lambda p: slice(None))):
        self._set_orf(orf)

        mask = sel(self.pairs)
        return 1.0 / np.sqrt(np.sum(self.orfs[mask]**2 / self.sigmas[mask]**2))

    def gw_mean(self):
        return np.array([10**(2.0 * self.params['gw_log10_A'])] * len(self.pairs))

    def _tracedot(self, orf, *args):
        ret = np.identity(len(self.Gamma))

        for i, (j,k) in zip(args[::2], args[1::2]):
            ret = np.dot(ret, self.FCFs[i] * self.Gamma * orf(self.psrs[j].pos, self.psrs[k].pos))

        return np.trace(ret)

    def gw_corr(self, orf=utils.hd_orf):
        Agw = 10 ** self.params['gw_log10_A']

        sigma = np.zeros((len(self.pairs), len(self.pairs)), 'd')

        for ij in tqdm(range(len(self.pairs))):
            i, j = self.pairs[ij]

            for kl in range(ij, len(self.pairs)):
                k, l = self.pairs[kl]

                if ij == kl:
                    sigma[ij, kl] = (self._tracedot(orf, i, (i, j), j, (j, i)) +
                                     Agw ** 4 * self._tracedot(orf, i, (i, j), j, (j, i), i, (i, j), j, (j, i)))

                elif i == k and j != l:
                    sigma[ij, kl] = (Agw ** 2 * self._tracedot(orf, i, (i, j), j, (j, l), l, (l, i)) +
                                     Agw ** 4 * self._tracedot(orf, i, (i, j), j, (j, i), i, (i, l), l, (l, i)))

                    # ijil -> ij il + ii jl + il ji
                    # iCGCj iCGCl -> A^4 (CGCG)^2 + A^2 iCGCj lCGCi + A^4 iCGCj iCGCl

                elif i != k and j == l:
                    sigma[ij, kl] = (Agw ** 2 * self._tracedot(orf, j, (j, k), k, (k, i), i, (i, j)) +
                                     Agw ** 4 * self._tracedot(orf, i, (i, j), j, (j, k), k, (k, j), j, (j, i)))

                    # ijkj -> ij kj + ik jj + ij jk
                    # iCGCj kCGCj -> A^4 (CGCG)^2 + A^2 jCGCk iCGCj + A^4 iCGCj kCGCj

                elif i != k and j != l and i != l and j != k:
                    sigma[ij, kl] = Agw ** 4 * (self._tracedot(orf, i, (i, j), j, (j, k), k, (k, l), l, (l, i)) +
                                                self._tracedot(orf, i, (i, j), j, (j, l), l, (l, k), k, (k, i)))

                    # ijkl -> ij kl + ik jl + il jk
                    # iCGCj kCGCl -> A^4 (CGCG)^2 + A^4 iCGCj lCGCk + A^4 iCGCj kCGCl

                sigma[ij, kl] = sigma[ij, kl] * self.sigmas[ij] ** 2 * self.sigmas[kl] ** 2 / orf(self.psrs[i].pos,
                                                                                                  self.psrs[
                                                                                                      j].pos) ** 2 / orf(
                    self.psrs[k].pos, self.psrs[l].pos) ** 2
                sigma[kl, ij] = sigma[ij, kl]

        return sigma

    def snr(self, orf=utils.hd_orf, sel=(lambda p: slice(None))):
        return self.os(orf, sel) / self.os_sigma(orf, sel)


def make_complex(vec):
    return vec[::2] + 1j * vec[1::2]

class OSShifts(OS):
    def __init__(self, psrs, pta, params):
        self.psrs, self.params = psrs, params
        self.pairs = [(i, j) for i in range(len(pta)) for j in range(i + 1, len(pta))]
        self.angles = np.array([np.arccos(np.dot(self.psrs[i].pos, self.psrs[j].pos)) for (i, j) in self.pairs])

        ys = [sm._residuals for sm in pta]
        Cs = [Cmat(sm, params) for sm in pta]

        Fgws = [sm['gw'].get_basis(params) for sm in pta]
        self.Gamma = pta[0]['gw'].get_phi({**params, 'gw_log10_A': 0});
        assert self.Gamma.ndim == 1

        FCys = [make_complex(C.solve(y, Fgw)) for C, y, Fgw in zip(Cs, ys, Fgws)]
        print(FCys[0])
        self.FCFs = [C.solve(Fgw, Fgw) for C, Fgw in zip(Cs, Fgws)]

        ts = [np.conj(FCys[i]) * (self.Gamma[::2] * FCys[j]) for (i, j) in self.pairs]
        bs = [np.trace(np.dot(self.FCFs[i] * self.Gamma, self.FCFs[j] * self.Gamma)) for (i, j) in self.pairs]

        self._rhos = np.array(ts) / np.array(bs)[:, np.newaxis]
        self.sigmas = 1.0 / np.sqrt(bs)

        self.set_shifts(None)

    def set_shifts(self, seed):
        ngw = self._rhos[0].shape[0]

        if seed is None:
            self.shifts = [np.ones(ngw) for i in range(len(self.psrs))]
        else:
            np.random.seed(seed)
            self.shifts = [np.exp(2 * math.pi * 1j * np.random.uniform(size=ngw)) for i in range(len(self.psrs))]

        self.allshifts = np.array([np.conj(self.shifts[i]) * self.shifts[j] for i, j in self.pairs])

    def set_freqs(self, k):
        ngw = self._rhos[0].shape[0]
        self.shifts = []
        for i in range(len(self.psrs)):
            self.shifts.append(np.zeros(ngw))
            self.shifts[-1][k] = 1

        self.allshifts = np.array([np.conj(self.shifts[i]) * self.shifts[j] for i, j in self.pairs])

    @property
    def rhos(self):
        return np.real(np.sum(self._rhos * self.allshifts, axis=1))
