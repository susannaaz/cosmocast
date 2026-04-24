import numpy as np
from copy import deepcopy
from itertools import combinations_with_replacement as cwr

from iso_theory import (
    compute_cls, knox_auto_cov, knox_cross_cov,
    make_cross_noise, _interp_to, _apply_cuts,
)


SO_FREQ  = [27.0,39.0,93.0, 145.0,225.0,280.0]
SO_CMB_FREQ = [93.0,145.0]
PK_FREQ = [100.0,143.0,217.0,353.0]
PK_CMB_FREQ = [100.0,143.0]

frequencies = {
    "LAT": ["027", "039", "093", "145", "225", "280"],
    "SAT": ["027", "039", "093", "145", "225", "280"],
    "PK": ["100", "143", "217", "353"]
}
SAT_pairs_cmb = ['SAT_093xSAT_093',
                'SAT_145xSAT_145',
                 'SAT_093xSAT_145',
                ]
LAT_pairs_cmb = [s.translate(str.maketrans('S','L')) for s in SAT_pairs_cmb]
                 
LAT_pairs_all = ["LAT_27xLAT_27",
               "LAT_39xLAT_39",
               "LAT_93xLAT_93",
               "LAT_145xLAT_145",
               "LAT_225xLAT_225",
               "LAT_280xLAT_280",
               "LAT_27xLAT_39",
               "LAT_93xLAT_145",
               "LAT_225xLAT_280"]

SAT_pairs_all = [s.translate(str.maketrans('L','S')) for s in LAT_pairs_all]

pk_pairs_cmb = ['PK_100xPK_100', 
                'PK_143xPK_143',
                'PK_100xPK_143']
pk_pairs_all = ["PK_{}xPK_{}".format(*cross) for cross in cwr(frequencies["PK"], 2)]


# even though some cases have weak theoretical motivations (or rather, they
# present currently unexplainable departures from LCDM), we consider all
# possible modes in our forecasts. We cannot rule out a priori the scenario
# that SO data might perform better on these modes only, and if that ends up
# being the case, we believe it would be a wasted opportunity to not
# investigate isocurvature modes at all.

ISO_TYPE_ARR = [None,'cdi','nid','niv']
CORR_TYPE_ARR = ['pcor','acor','ucor']
# in the correlated/fully anti-correlated, we consider:
# we vary only P(1)  II  and fix P(2)  II assuming nII = nRR


# ── noise dict building ─────────────────────────────────────────────────────

def add_case(experiment1,dict_init = None,yrs = 1,sens_mode = 1, f_mode = 0, lmax = 10_000, fsky = 0.1, dell = 10, cuts = [30,300]):
    '''make f_mode negative if there is no such thing in the experiment for LAT
        the function needs to be called any time you want to add a case'''
    new_dict = deepcopy(dict_init)
    if not dict_init:
        new_dict = {}
    if experiment1 == 'SAT':
        key = "{}_y{:.0f}_sm{:.0f}fm{:.0f}".format(experiment1,yrs ,sens_mode,f_mode)
        new_dict[key]= {
        'yrs' : yrs,
        'sens_mode' : sens_mode,
        'f_mode' : f_mode,
        'fsky' : fsky,
        'lmax' : lmax,
        'dell' : dell,
        'cuts': cuts,
                        }
    elif experiment1 == 'LAT' or experiment1 == 'LAT_pol':
        key = "{}_y{:.0f}_sm{:.0f}".format(experiment1, yrs,sens_mode)
        new_dict[key]= {
        'yrs' : yrs,
        'sens_mode' : sens_mode,
        'fsky' : fsky,
        'lmax' : lmax,
        'dell' : dell,
        'cuts': cuts,
            }
    return new_dict


def make_nz_dict_from_array(nz_array,freq_dict,experiment):
    '''
    noise is for freqxfreq; constructed with 'frequencies' dict in mind
    '''
    nz_dict = {}
    for i,name in enumerate(freq_dict[experiment]):
        # `frequencies[...]` are often zero-padded (e.g. "027"), while some
        # pair keys use the non-padded form (e.g. "SAT_27xSAT_27"). To stay
        # compatible with both conventions, store both spellings.
        key_padded = 'f{}'.format(name) if len(name) != 2 else 'f0{}'.format(name)
        nz_dict[key_padded] = nz_array[i]
        try:
            key_unpadded = 'f{}'.format(int(name))
            nz_dict.setdefault(key_unpadded, nz_array[i])
        except Exception:
            pass
    return nz_dict


def make_planck_noise(full_noise_dict, lmin_tt=2, lmax_tt=2000, lmin_ee=2, lmax_ee=2000, dell_tt=10, dell_ee=10):
    """
    IMPORTANT NOTE: in order to make the te covariance, tt and ee need to have the same size (if building from knox).
    after building, the last columns can be struck, according to preference of ell-range!
    'normal' cov matrices can have non-square TE, but a Gaussian TE cannot, by definition.
    """
    ell_pk_tt = np.arange(lmin_tt,lmax_tt, dell_tt)
    ell_pk_ee = np.arange(lmin_ee,lmax_ee, dell_ee)

    sigma = {"PK_100xPK_100":0.,
            "PK_143xPK_143":0.,
            "PK_217xPK_217":0.,
            "PK_353xPK_353":0.,
            }
    sigma_pol = deepcopy(sigma)


    # converted temperature noise level from Table 4 of 1807.06205 to micro-Kelvin*arcmin
    sigma["PK_100xPK_100"] = 77.4
    sigma["PK_143xPK_143"] = 33.0
    sigma["PK_217xPK_217"] = 46.80
    sigma["PK_353xPK_353"] = 153.6

    # converted polarization noise level from Table 4 of 1807.06205 to micro-Kelvin*arcmin
    sigma_pol["PK_100xPK_100"] = 117.6
    sigma_pol["PK_143xPK_143"] = 70.2
    sigma_pol["PK_217xPK_217"] = 105.0
    sigma_pol["PK_353xPK_353"] = 438.6

    ttnz_PK = np.zeros((4,len(ell_pk_tt)))
    eenz_PK = np.zeros((4,len(ell_pk_ee)))

    for i,f_pair in enumerate(sigma.keys()):
        sigma_rad = np.deg2rad(sigma[f_pair]/ 60) 
        ttnz_PK[i] = ell_pk_tt * 0 + sigma_rad**2
        sigma_pol_rad = np.deg2rad(sigma_pol[f_pair]/ 60)
        eenz_PK[i] = ell_pk_ee * 0 + sigma_pol_rad**2

    full_noise_dict['PK'] = { 'lmax':lmax_tt,
                              'dell':dell_tt,
                              'fsky':1.0,
                              'cuts':[lmin_tt,lmax_tt],
                              'nz_dict':  make_nz_dict_from_array( ttnz_PK,frequencies,'PK'),
                              'ell_nz':ell_pk_tt,
                            }
    full_noise_dict['PK_pol'] = { 'lmax':lmax_ee,
                                  'dell':dell_ee,
                                  'fsky':1.0,
                                  'cuts':[lmin_ee,lmax_ee],
                                  'nz_dict':  make_nz_dict_from_array( eenz_PK,frequencies,'PK'),
                                  'ell_nz' : ell_pk_ee,
                                    }
    full_noise_dict['PK_cross'] = { 'lmax':lmax_ee,
                                    'dell':dell_ee,
                                    'fsky':1.0,
                                    'cuts':[lmin_ee,lmax_ee],
                                    'nz_dict': None,
                                    'ell_nz' : ell_pk_ee,
                                    }
    return full_noise_dict


# ── likelihood data assembly ─────────────────────────────────────────────────

# you have an array with the noise spectra and the params you used to create them
# for each separate experiment: LAT, SAT, Planck;
# it is time to create what will become your likelihood.
# ideally, you would store just the spectra, and store other metadata separately, if need be

def build_full_lik_data(req_spec, setup, full_noise_dict, cmb_theo_dict):
    '''req_spec is a dict that tells you first what experiments you will want in your likelihood,
    and for each what spectra and what frequencies.
    in the way this is currently set up (which can be subject to change),
    your full_noise_dict is what sets the tone for the cuts, the binning, etc.'''

    full_lik_data = {'metadata':{}, 
                    'data':{},
                    }
    #input the power spectra that you desire:
    for exp_key in req_spec:
        check_cross = False
        if exp_key in setup.keys():
            setup_exp = setup[exp_key]
        
        if exp_key == 'SAT':
            case_key = "{}_y{:.0f}_sm{:.0f}fm{:.0f}".format(exp_key,setup_exp['yrs'] ,setup_exp['sens_mode'],setup_exp['f_mode'])
            c_ell_theo = cmb_theo_dict['EE']
            cell_type = 'EE'
        elif exp_key == 'LAT' :
            case_key = "{}_y{:.0f}_sm{:.0f}".format(exp_key,setup_exp['yrs'] ,setup_exp['sens_mode'],setup_exp['f_mode'])
            c_ell_theo = cmb_theo_dict['TT']
            cell_type = 'TT'

        elif exp_key == 'LAT_pol' :
            case_key = "{}_y{:.0f}_sm{:.0f}".format(exp_key,setup_exp['yrs'] ,setup_exp['sens_mode'],setup_exp['f_mode'])
            c_ell_theo = cmb_theo_dict['EE']
            cell_type = 'EE'

        elif exp_key == 'LAT_cross':
            check_cross = True
            case_key = "{}_y{:.0f}_sm{:.0f}".format(exp_key,setup_exp['yrs'] ,setup_exp['sens_mode'],setup_exp['f_mode'])
            c_ell_theo = cmb_theo_dict['TE']
            cell_type = 'TE'
          
        elif exp_key == 'PK' :
            case_key = exp_key
            c_ell_theo = cmb_theo_dict['TT']
            cell_type = 'TT'

        
        elif exp_key == 'PK_pol' :
            case_key = exp_key
            c_ell_theo = cmb_theo_dict['EE']
            cell_type = 'EE'

            
        elif exp_key == 'PK_cross':
            check_cross = True
            case_key = exp_key
            c_ell_theo = cmb_theo_dict['TE']
            cell_type = 'TE'

        pairs_current = req_spec[exp_key]
        if not check_cross:
            nz_dict = deepcopy(full_noise_dict[case_key]['nz_dict'])
        else:
            nz_dict = {}

        meta = deepcopy(full_noise_dict[case_key])
        meta.pop('nz_dict')
        meta.pop('ell_nz')
        full_lik_data['metadata'][exp_key] = meta
        full_lik_data['metadata'][exp_key]['cell_type'] = cell_type
        full_lik_data['data'][exp_key] = {}
        for spec_key in pairs_current:

            left, right = spec_key.split('x')
            f1 = left.split('_')[-1]
            f2 = right.split('_')[-1]
            
            ell_new = full_noise_dict[case_key]['ell_nz']
            ell_old = cmb_theo_dict['ell']
            cuts = full_noise_dict[case_key]['cuts']
            ell_old_cut, cell_cmb_cut = _apply_cuts(c_ell_theo, cmb_theo_dict['ell'], cuts)
            cell_cmb_cut = _interp_to(ell_new, ell_old_cut,cell_cmb_cut)
            ell_new_cut, cell_cmb_cut = _apply_cuts(cell_cmb_cut, ell_new, cuts)
            if not check_cross:
                if f1 == f2:
                    nz_arr = nz_dict['f{}'.format(f1)]
                    _, nz_arr_cut = _apply_cuts(nz_arr, ell_new, cuts)
                else:
                    nz_arr = make_cross_noise(nz_dict['f{}'.format(f1)],nz_dict['f{}'.format(f2)])
                    _, nz_arr_cut = _apply_cuts(nz_arr, ell_new, cuts)
            else:
                nz_arr_cut = cell_cmb_cut*0
            cell_mock = cell_cmb_cut + nz_arr_cut
            full_lik_data['data'][exp_key][spec_key] = {'c_ell' : cell_mock, 'ell': ell_new_cut}

    return full_lik_data


# ── covariance construction ──────────────────────────────────────────────────

def build_full_lik_cov(full_lik_data, setup, cases,
                       knox_auto_cov_func=knox_auto_cov,
                       knox_cross_cov_func=knox_cross_cov):

    auto_blocks = {'SAT', 'LAT', 'LAT_pol', 'PK', 'PK_pol'}
    cross_block_map = {
        'LAT_cross': ('LAT', 'LAT_pol'),
        'PK_cross': ('PK', 'PK_pol'),
    }


    full_lik_cov = {'metadata': deepcopy(full_lik_data['metadata']),
                    'data':{}}

    for exp_key, spec_dict in full_lik_data['data'].items():
        dell_curr, fsky_curr = full_lik_data['metadata'][exp_key]['dell'],full_lik_data['metadata'][exp_key]['fsky']

        full_lik_cov['data'][exp_key] = {}

        if exp_key in auto_blocks:
            for spec_key, spec_info in spec_dict.items():
                c_ell = np.asarray(spec_info['c_ell'])
                ell = np.asarray(spec_info['ell'])

                cov = knox_auto_cov_func(
                    c_ell=c_ell,
                    ell=ell,
                    delta_ell=dell_curr,
                    fsky=fsky_curr
                )

                full_lik_cov['data'][exp_key][spec_key] = {
                    'cov': cov,
                    'ell': ell
                }


        elif exp_key in cross_block_map:
            tt_block, ee_block = cross_block_map[exp_key]

            if tt_block in full_lik_data['data'] and ee_block in full_lik_data['data']:
                for spec_key, spec_info in spec_dict.items():
                    if spec_key in full_lik_data['data'][tt_block] and spec_key in full_lik_data['data'][ee_block]:
                        c_cross = np.asarray(spec_info['c_ell'])
                        ell_cross = np.asarray(spec_info['ell'])

                        c_1 = np.asarray(full_lik_data['data'][tt_block][spec_key]['c_ell'])
                        c_2 = np.asarray(full_lik_data['data'][ee_block][spec_key]['c_ell'])

                        cov = knox_cross_cov_func(
                            c_cross=c_cross,
                            c_1=c_1,
                            c_2=c_2,
                            ell=ell_cross,
                            delta_ell=dell_curr,
                            fsky=fsky_curr
                        )

                        full_lik_cov['data'][exp_key][spec_key] = {
                            'cov': cov,
                            'ell': ell_cross
                        }

    return full_lik_cov
