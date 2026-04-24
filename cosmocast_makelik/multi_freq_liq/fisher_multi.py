import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable


@dataclass
class SpectrumBand:
    """
    One (experiment, frequency-pair, spectrum-type) data stream, 
    trimmed to the configured ell cuts.

    Attributes
    ----------
    exp_key   : which experiment block this came from
    channel   : frequency-pair key, e.g. 'SAT_093xSAT_093'
    cell_type : 'TT', 'EE', or 'TE'
    ell       : bin-centre multipoles after cuts,  shape (n_ell,)
    dell      : bin width Δℓ  (scalar, from metadata)
    fsky      : sky fraction   (scalar, from metadata)
    cov       : diagonal of the Gaussian power-spectrum covariance,
                shape (n_ell,).  Each entry is Var(Ĉ^XY_ℓ) at that bin.
    """
    exp_key:   str
    channel:   str
    cell_type: str
    ell:       np.ndarray
    dell:      int
    fsky:      float
    cov:       np.ndarray


@dataclass
class FisherResult:
    F:          np.ndarray
    Cov_params: np.ndarray
    sigma:      np.ndarray
    dC:         list
    bands:      list
    param_list: list[str]
    metadata:   dict = field(default_factory=dict)


    def with_prior(self, priors: dict[str, float]) -> "FisherResult":
        unknown = set(priors) - set(self.param_list)
        if unknown:
            raise KeyError(f"Prior specified for unknown parameters: {unknown}")
        F_prior = self.F.copy()
        for param, sigma_prior in priors.items():
            i = self.param_list.index(param)
            F_prior[i, i] += 1.0 / sigma_prior**2
        Cov_prior = np.linalg.pinv(F_prior)
        return FisherResult(
            F=F_prior, Cov_params=Cov_prior,
            sigma=np.sqrt(np.diag(Cov_prior)),
            dC=self.dC, bands=self.bands, param_list=self.param_list,
            metadata=self.metadata,
        )

    def combine(self, other: "FisherResult") -> "FisherResult":
        if self.param_list != other.param_list:
            raise ValueError("param_lists must match to combine.")
        F_combined = self.F + other.F
        Cov_combined = np.linalg.pinv(F_combined)
        metadata = {}
        if isinstance(self.metadata, dict):
            metadata.update(self.metadata)
        if isinstance(other.metadata, dict):
            metadata.update(other.metadata)
        return FisherResult(
            F=F_combined, Cov_params=Cov_combined,
            sigma=np.sqrt(np.diag(Cov_combined)),
            dC=self.dC + other.dC,
            bands=self.bands + other.bands,
            param_list=self.param_list,
            metadata=metadata,
        )

    def _centers(self, scaled_params: set = frozenset()) -> np.ndarray:
        """Fiducial values in display/differentiation space (scaled where needed)."""
        return np.array(
            [1e10 * self._theta0[p] if p in scaled_params else self._theta0[p]
             for p in self.param_list],
            dtype=float,
        )

    def _correlation_matrix(self) -> np.ndarray:
        den  = np.outer(self.sigma, self.sigma)
        Corr = np.divide(self.Cov_params, den,
                         out=np.zeros_like(self.Cov_params), where=den > 0)
        np.fill_diagonal(Corr, 1.0)
        return Corr

    def _latex_labels(self, label_map: dict[str, str]) -> list[str]:
        return [label_map.get(p, p) for p in self.param_list]

    def summary_table(
        self,
        theta0:        dict,
        scaled_params: set = frozenset(),
    ) -> "pd.DataFrame":
        """
        Return a DataFrame with columns Fiducial / sigma / S·N⁻¹.
        Rows indexed by param_list; scaled params shown in ×1e10 space.
        """
        import pandas as pd
        centers = np.array(
            [1e10 * theta0[p] if p in scaled_params else theta0[p]
             for p in self.param_list],
            dtype=float,
        )
        return pd.DataFrame(
            {
                "Fiducial": centers,
                "sigma":    self.sigma,
                "S/N":      np.divide(
                    centers, self.sigma,
                    out=np.full_like(centers, np.nan),
                    where=self.sigma > 0,
                ),
            },
            index=self.param_list,
        )

    def plot_correlation(
        self,
        exp_name:  str  = "",
        save_path: str  = None,
        ax=None,
    ):
        Corr = self._correlation_matrix()
        n    = len(self.param_list)

        if ax is None:
            fig, ax = plt.subplots(figsize=(0.7 * n + 3, 0.7 * n + 3))
        else:
            fig = ax.figure

        im = ax.imshow(Corr, vmin=-1, vmax=1)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(self.param_list, rotation=45, ha="right")
        ax.set_yticklabels(self.param_list)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
        ax.set_title(f"Correlation matrix {exp_name}")
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path)
        return fig, ax

    def plot_triangle(
        self,
        theta0:        dict,
        label_map:     dict[str, str] = None,
        scaled_params: set            = frozenset(),
        exp_name:      str            = "",
        save_path:     str            = None,
    ):
        from getdist.gaussian_mixtures import GaussianND
        from getdist import plots

        label_map  = label_map or {}
        centers    = np.array(
            [1e10 * theta0[p] if p in scaled_params else theta0[p]
             for p in self.param_list],
            dtype=float,
        )
        labels = [label_map.get(p, p) for p in self.param_list]

        gauss = GaussianND(centers, self.Cov_params,
                           names=self.param_list, labels=labels)
        g = plots.get_subplot_plotter()
        g.triangle_plot([gauss], self.param_list, filled=True)
        plt.suptitle(f"{exp_name} Fisher")

        if save_path is not None:
            plt.savefig(save_path)
        return g

    def save_summary_table(
    self,
    theta0:        dict,
    scaled_params: set  = frozenset(),
    exp_name:      str  = "",
    title:         str  = None,
    save_path:     str  = None,
    ):
        df  = self.summary_table(theta0, scaled_params)
        fig, ax = plt.subplots(figsize=(8, 0.6 * len(df) + 1))
        ax.axis("off")
        table = ax.table(
            cellText=np.round(df.values, 4),
            colLabels=df.columns,
            rowLabels=df.index,
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
    
        if title is not None:
            fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    
        fig.tight_layout()
        path = save_path or f"fisher_summary_{exp_name}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return df



def parse_spectrum_bands(full_lik_cov, exp_keys) -> list[SpectrumBand]:
    """
    Parse every requested experiment into a flat list of SpectrumBand objects.

    Parameters
    ----------
    full_lik_cov : dict
        The input dictionary (metadata + data).
    exp_keys : str or list[str]
        Experiment keys to include, e.g. ['SAT', 'PK', 'PK_pol', 'PK_cross'].

    Returns
    -------
    list[SpectrumBand]
        One entry per (exp_key × channel_pair), ell-trimmed to configured cuts.
    """
    if isinstance(exp_keys, str):
        exp_keys = [exp_keys]

    bands = []
    for exp_key in exp_keys:
        meta          = full_lik_cov["metadata"][exp_key]
        cell_type     = meta["cell_type"]
        fsky          = meta["fsky"]
        dell          = meta["dell"]
        lmin, lmax    = meta["cuts"]

        for channel, ch_dict in full_lik_cov["data"][exp_key].items():
            ell  = np.asarray(ch_dict["ell"], dtype=int)
            cov  = np.asarray(ch_dict["cov"], dtype=float)
            mask = (ell >= lmin) & (ell <= lmax)

            bands.append(SpectrumBand(
                exp_key   = exp_key,
                channel   = channel,
                cell_type = cell_type,
                ell       = ell[mask],
                dell      = dell,
                fsky      = fsky,
                cov       = cov[mask],
            ))

    return bands


def parse_planck_lite_bands(
    planck_data: dict,
    cell_types:  list[str] = None,
) -> list[SpectrumBand]:
    if cell_types is None:
        cell_types = list(planck_data.keys())

    bands = []
    for cell_type in cell_types:
        d     = planck_data[cell_type]
        ell   = np.asarray(d["ell"],   dtype=float)
        bell  = np.asarray(d["b_ell"], dtype=float)

        # err IS the variance Var(C_hat_ell) — not sigma, not noise level
        cov = np.asarray(d["err"], dtype=float)

        bands.append(SpectrumBand(
            exp_key   = "PlanckLite",
            channel   = f"PlanckLite_{cell_type}",
            cell_type = cell_type,
            ell       = ell.astype(int),
            dell      = bell,
            fsky      = 1.0,
            cov       = cov,
        ))

    return bands


# ── Cl evaluation on bands ───────────────────────────────────────────────────

def eval_cls_on_bands(bands: list[SpectrumBand], raw_cls: dict) -> list[np.ndarray]:
    """
    Sample theory Cls at each band's ell grid.

    Parameters
    ----------
    bands   : list[SpectrumBand]  (from parse_spectrum_bands)
    raw_cls : dict  {cell_type: array indexed by integer ell}

    Returns
    -------
    list of (n_ell,) arrays, one per band
    """
    return [np.asarray(raw_cls[b.cell_type])[b.ell] for b in bands]


# ── derivatives ──────────────────────────────────────────────────────────────

def _step_for(param: str, steps: dict, scaled_params: set) -> tuple[float, bool]:
    """
    Return (step_size_in_differentiation_space, is_scaled).

    For scaled params the step is in the ×1e10 space; raw step is stored in `steps`.
    """
    if param not in steps:
        raise KeyError(f"No step size provided for parameter '{param}'")
    return steps[param], (param in scaled_params)


def _perturb(theta0: dict, param: str, delta: float, scaled_params: set) -> dict:
    """Return a copy of theta0 with one parameter perturbed by delta."""
    th = dict(theta0)
    if param in scaled_params:
        th[param] = (1e10 * theta0[param] + delta) * 1e-10
    else:
        th[param] = theta0[param] + delta
    return th


def compute_dC_bands(
    theta0:        dict,
    param:         str,
    step:          float,
    bands:         list,           # list[SpectrumBand]
    compute_cls:   Callable,
    scaled_params: set  = frozenset(),
    ell_max:       int  = 10_000,
) -> list:
    """
    Central-difference dC_ℓ/d(param) for every band.

    Returns
    -------
    list of (n_ell_b,) arrays, one per band
    """
    theta_hi = _perturb(theta0, param, +step, scaled_params)
    theta_lo = _perturb(theta0, param, -step, scaled_params)

    cls_hi = compute_cls(**theta_hi, lmax=ell_max)
    cls_lo = compute_cls(**theta_lo, lmax=ell_max)

    return [
        (np.asarray(cls_hi[b.cell_type])[b.ell]
         - np.asarray(cls_lo[b.cell_type])[b.ell]) / (2.0 * step)
        for b in bands
    ]




# ── Fisher forecast ──────────────────────────────────────────────────────────

# rationale for Fisher:
# - func takes in full_lik_cov + specification of which exp_key (= experiment key) I want in the covariance
#       - smaller func makes the cov by just stringing things together (make np.diag and stack if
#         length of element in cov_ask = 3, which indicates TT, TE EE covariance, and cov is stacked
#         as [[TT,TE],[TE,EE]], where each denotes a diagonal matrix; if only one experiment is
#         requested, then just put all the spectra in one array and do np.diag.)
#
# - find the type in the metadata (full_lik_data['metadata'][exp_key]['cell_type'];
#   options: 'TT','TE', 'EE')
#
# - make dcls (use central diff, then stack just like cov)
#
# - run the fisher algo, which is just an implementation of:
#       F_{ij} = sum_ell (2ell+1)*fsky*Tr(C_ell^-1 dC/dtheta_i C_ell^-1 dC/dtheta_j)

def fisher_forecast(
    theta0:        dict,
    param_list:    list[str],
    bands:         list,           # list[SpectrumBand]  (from parse_spectrum_bands
                                   # OR any external source with .ell, .cov, .cell_type)
    compute_cls:   Callable,
    steps:         dict,           # {param: step_size}  in differentiation space
    scaled_params: set   = frozenset(),
    ell_max:       int   = 10_000,
    use_pinv:      bool  = False,
) -> FisherResult:
    """
    Compute the Fisher matrix via the diagonal Gaussian covariance formula:

        F_αβ = Σ_{bands} Σ_ℓ  (dC_α[ℓ] · dC_β[ℓ]) / Var(Ĉ_ℓ)

    where band.cov already encodes  Var(Ĉ_ℓ) = 2C_ℓ²/[(2ℓ+1)Δℓ f_sky],
    so no additional (2ℓ+1) / f_sky weighting is applied here.

    Parameters
    ----------
    theta0        : fiducial parameter dict
    param_list    : parameters to forecast, defines row/column ordering of F
    bands         : iterable of objects with  .cell_type, .ell, .cov  attributes
                    Can be SpectrumBand objects from parse_spectrum_bands(), or
                    any external covariance source (Planck_Lite, etc.) as long as
                    it exposes the same interface.
    compute_cls   : callable(**theta, lmax=int) -> {cell_type: Cl_array}
    steps         : step sizes, keyed by param name
    scaled_params : params whose fiducial values are multiplied by 1e10 before
                    differencing (e.g. primordial power spectrum amplitudes)
    ell_max       : passed to compute_cls
    use_pinv      : use pseudo-inverse instead of inv when inverting F

    Returns
    -------
    FisherResult
    """
    # ── 1. Derivatives ───────────────────────────────────────────────────────
    dC = []
    for param in param_list:
        step, is_scaled = _step_for(param, steps, scaled_params)
        dC_p = compute_dC_bands(
            theta0        = theta0,
            param         = param,
            step          = step,
            bands         = bands,
            compute_cls   = compute_cls,
            scaled_params = scaled_params,
            ell_max       = ell_max,
        )
        dC.append(dC_p)   # dC[i][b] has shape (n_ell_b,)

    # ── 2. Fisher matrix ─────────────────────────────────────────────────────
    # band.cov is Var(Ĉ_ℓ), so the weight is simply 1/Var.
    # Summing over all bands and all ell bins within each band.
    npar = len(param_list)
    F = np.zeros((npar, npar), dtype=float)

    for i in range(npar):
        for j in range(i, npar):
            s = sum(
                np.sum(dC[i][b] * dC[j][b] / band.cov)
                for b, band in enumerate(bands)
            )
            F[i, j] = s
            F[j, i] = s

    # ── 3. Parameter covariance ──────────────────────────────────────────────
    inv = np.linalg.pinv if use_pinv else np.linalg.inv
    Cov_params = inv(F)
    sigma      = np.sqrt(np.diag(Cov_params))

    return FisherResult(
        F          = F,
        Cov_params = Cov_params,
        sigma      = sigma,
        dC         = dC,
        bands      = list(bands),
        param_list = list(param_list),
    )


# ── plot/save convenience ────────────────────────────────────────────────────

def plot_save_suite(result, EXP_NAME, theta0, scaled_params,
                   LATEX = {
                            "omega_b":   r"\omega_b",
                            "omega_cdm": r"\omega_{\rm cdm}",
                            "h":         r"h",
                            "tau_reio":  r"\tau",
                            "P_RR_1":    r"10^{10}P_{RR,1}",
                            "P_RR_2":    r"10^{10}P_{RR,2}",
                            "P_II_1":    r"10^{10}P_{II,1}",
                            "P_II_2":    r"10^{10}P_{II,2}",
                        },
                   ):
    table_savepath = "images/032226_fisher/sum_table_{}.png".format(EXP_NAME)
    corrmat_savepath = "images/032226_fisher/corr_matrix_{}.png".format(EXP_NAME)
    triangle_savepath = "images/032226_fisher/triangle_plot_{}.png".format(EXP_NAME)
    
    result_now = deepcopy(result)
    
    df = result_now.save_summary_table(
        theta0=theta0, scaled_params=scaled_params,
        exp_name=EXP_NAME, save_path=table_savepath,
    )
    
    result_now.plot_correlation(exp_name=EXP_NAME, save_path=corrmat_savepath)
    
    result_now.plot_triangle(
        theta0=theta0, label_map=LATEX,
        scaled_params=scaled_params,
        exp_name=EXP_NAME, save_path=triangle_savepath,
    )
    return df
