import numpy as np
try:
    from classy import Class
except ModuleNotFoundError:  # optional dependency (CLASS)
    Class = None

def make_isocur_params(
    lmax=10_000,
    lensing=True,
    A_s=2.1e-9,
    n_s=0.9665,
    tau_reio=0.056,
    omega_b=0.02237,
    omega_cdm=0.1200,
    h=0.6736,
    # isocurvature controls (for one chosen extra mode; extend as needed)
    iso_mode='cdi',          # 'cdi','bi','nid','niv' or None
    k1= 0.002,
    k2=0.1,
    P_RR_1 = 2.3e-9,
    P_RR_2 = 2.3e-9,
    P_II_1 = 1.0e-11,
    P_II_2 = 1.0e-11,
    #sign of P_RI_1 should indicate whether correlated or anti-correlated
    P_RI_1 = 1.0e-13,
    #always positive
    P_RI_2 = 1.0e-13,
    high_accuracy = False
):

    params = {
        'output': 'tCl,pCl,lCl,mPk',                 # <- include lCl so lensing can be computed
        'lensing': 'yes' if lensing else 'no',
        'l_max_scalars': lmax, 'tau_reio': tau_reio,
        'omega_b': omega_b, 'omega_cdm': omega_cdm, 'h': h,
        'modes': 's',
        'ic': 'ad',
        'P_k_ini type': 'two_scales',
        'k1': k1,
        'k2': k2,
        'P_{RR}^1': P_RR_1,
        'P_{RR}^2': P_RR_2
        # NOTE: intentionally NOT setting 'l_max_lss' to avoid "not read" errors on some builds.
    }

    if iso_mode is not None:
        params['ic'] = f'ad,{iso_mode}'
        params['P_{II}^1'] = P_II_1
        params['P_{II}^2'] = P_II_2
        params['P_{RI}^1'] = P_RI_1
        params['|P_{RI}^2|'] = P_RI_2
        high_accuracy = {
                    'N_ncdm': 1,
                    'm_ncdm': 0.06,
                    'N_ur': 2.0308,
                    'T_cmb': 2.7255,
                    'YHe': 'BBN',
                    'non linear':'hmcode',
                    'hmcode_version': '2020',
                    'recombination': 'HyRec',
                    'lensing':'yes',
                    'output': 'tCl, pCl, lCl, mPk',
                    'modes': 's',
                    'l_max_scalars': 9500,
                    'delta_l_max': 1800,
                    'P_k_max_h/Mpc': 100.0,
                    'l_logstep': 1.025,
                    'l_linstep': 20,
                    'perturbations_sampling_stepsize': 0.05,
                    'l_switch_limber': 30.0,
                    'hyper_sampling_flat': 32.0,
                    'l_max_g': 40,
                    'l_max_ur': 35,
                    'l_max_pol_g': 60,
                    'ur_fluid_approximation': 2,
                    'ur_fluid_trigger_tau_over_tau_k': 130.0,
                    'radiation_streaming_approximation': 2,
                    'radiation_streaming_trigger_tau_over_tau_k': 240.0,
                    'hyper_flat_approximation_nu': 7000.0,
                    'transfer_neglect_delta_k_S_t0': 0.17,
                    'transfer_neglect_delta_k_S_t1': 0.05,
                    'transfer_neglect_delta_k_S_t2': 0.17,
                    'transfer_neglect_delta_k_S_e': 0.17,
                    'accurate_lensing': 1,
                    'start_small_k_at_tau_c_over_tau_h': 0.0004,
                    'start_large_k_at_tau_h_over_tau_k': 0.05,
                    'tight_coupling_trigger_tau_c_over_tau_h': 0.005,
                    'tight_coupling_trigger_tau_c_over_tau_k': 0.008,
                    'start_sources_at_tau_c_over_tau_h': 0.006,
                    'l_max_ncdm': 30,
                    'tol_ncdm_synchronous': 1.0e-06
                    }
    #if high_accuracy:
        #params.update(high_accuracy)

    return params

def compute_cls(
    lmax=10_000,
    lensing=True,
    A_s=2.1e-9,
    n_s=0.9665,
    tau_reio=0.056,
    omega_b=0.02237,
    omega_cdm=0.1200,
    h=0.6736,
    # isocurvature controls (for one chosen extra mode; extend as needed)
    iso_mode='cdi',          # 'cdi','bi','nid','niv' or None
    k1= 0.002,
    k2=0.1,
    P_RR_1 = 2.3e-9,
    P_RR_2 = 2.3e-9,
    P_II_1 = 1.0e-11,
    P_II_2 = 1.0e-11,
    #sign of P_RI_1 should indicate whether correlated or anti-correlated
    P_RI_1 = 1.0e-13,
    #always positive
    P_RI_2 = 1.0e-13,

):
    if Class is None:
        raise ModuleNotFoundError(
            "Missing optional dependency `classy` (CLASS). "
            "Install CLASS/classy to enable compute_cls()."
        )
    params = make_isocur_params(
        lmax=lmax,
        lensing=lensing,
        A_s=A_s,
        n_s=n_s,
        tau_reio=tau_reio,
        omega_b= omega_b,
        omega_cdm= omega_cdm,
        h= h ,
        # isocurvature controls (for one chosen extra mode; extend as needed)
        iso_mode= iso_mode,          # 'cdi','bi','nid','niv' or None
        k1= k1,
        k2= k2,
        P_RR_1 = P_RR_1,
        P_RR_2 = P_RR_2,
        P_II_1 = P_II_1,
        P_II_2 = P_II_2,
        #sign of P_RI_1 should indicate whether correlated or anti-correlated
        P_RI_1 = P_RI_1,
        #always positive
        P_RI_2 = P_RI_2,
    )
        

    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    # lensed spectra in K^2 -> convert to μK^2
    #NOTE: classy returns power spectrum in dimensionless units, so you need to 
    # account for that when converting...
    T_CMB_muK = 2.7255e6  # μK
    
    cl  = cosmo.lensed_cl(lmax)
    ell = cl['ell'][1:]
    TT  = cl['tt'][1:] * T_CMB_muK**2
    EE  = cl['ee'][1:] * T_CMB_muK**2
    BB  = cl['bb'][1:] * T_CMB_muK**2
    TE  = cl['te'][1:] * T_CMB_muK**2
    cosmo.struct_cleanup()
    cosmo.empty()
    return {'ell': ell, 'TT': TT, 'EE': EE, 'BB': BB, 'TE': TE}

def knox_auto_cov(c_ell, ell, delta_ell, fsky):
    return 2/(2*ell + 1)/delta_ell /fsky* c_ell**2

def knox_cross_cov(c_cross, c_1, c_2, ell, delta_ell, fsky):
    return 1/(2*ell + 1)/delta_ell /fsky * (c_cross**2 + c_1*c_2)

def make_cross_noise(cell_nz_1,cell_nz_2):
    return np.sqrt(cell_nz_1 * cell_nz_2)


def _interp_to(x_new, x_old, y_old):
    return np.interp(x_new, x_old, y_old)

def _apply_cuts(powspec,ells,cuts):
    lmin,lmax = cuts
    new_ell = ells[(ells >= lmin) & (ells <= lmax)]
    new_spec = powspec[(ells >= lmin) & (ells <= lmax)]
    return new_ell,new_spec
