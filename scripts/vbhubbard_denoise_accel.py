from triqs.gf import *
from triqs.gf import gf_fnt
import importlib.util
from triqs.gf.dlr_crm_dyson_solver import minimize_dyson
from triqs.operators import *
from h5 import *
from triqs_cthyb import Solver
from triqs_cthyb.tail_fit import tail_fit as cthyb_tail_fit
import numpy as np
from triqs.atom_diag import trace_rho_op
from triqs.operators import c, c_dag, n
import triqs.utility.mpi as mpi
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
import torch
from scipy.integrate import simpson
from pathlib import Path
from gtaudenoise.utilities.utils import *
from gtaudenoise.utilities.hankel_utils import *
from gtaudenoise.clean.cleaner import *
from gtaudenoise.models import AuxModel
np.set_printoptions(threshold=np.inf)
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
set_utils_device(device)


mus = [0.616, 0.5, 0.43, 0.27]

Path("vbhubbard_data").mkdir(parents=True, exist_ok=True)

#### Just generating some example data for normalization later within the model ####
beta = 200.0
N_tau = 2**10 + 1
N_ws = 1001
ws_max = 10
N = 2**15
set_beta(beta, N_tau)
set_ws(N_ws, ws_max)
giws, gtaus, Aws = sample_Aw(N, num_gauss=24, wr_cutoff=8)
torch.save({"gtaus_train": gtaus, "giws_train": giws, "Aws_train": Aws,
             "beta": beta, "ws_max": ws_max, "N_tau": N_tau, "N_ws": N_ws, 
             "num_gauss": 24, "wr_cutoff": 8}, "sample_data.pt")


for mu in mus:
    # Model params
    t = 0.25
    tp = -0.3*t
    U = 10.*t
    beta = 200./(4*t)

    ## Initialize the denoiser model and impurity solver
    if mpi.is_master_node():
        HD = AuxSpecDenoise(device, model_path="../model_weights/beta200_model.pth", data_path="sample_data.pt", beta=beta, ntau=2**10+1)

    # Get the dispersion over the BZ, cause we slicing it up
    k_linear = np.linspace(-np.pi, np.pi, 1000, endpoint=False)
    kx, ky = np.meshgrid(k_linear, k_linear)
    epsk = -2 * t * (np.cos(kx) + np.cos(ky)) - 4 * tp * np.cos(kx) * np.cos(ky)


    # Calculate energies inside and outside central patch
    in_central_patch = (np.abs(kx) < np.pi/np.sqrt(2)) & (np.abs(ky) < np.pi/np.sqrt(2))
    n_bins = 50
    energies, epsilon, rho, delta = {},{}, {}, {}
    energies['even'] = np.extract(in_central_patch, epsk)
    energies['odd'] = np.extract(np.invert(in_central_patch), epsk)

    ### Bin the energies in the BZ and calculate the prob dens of being at certain energy as well as energy values by averaging
    for patch in ['even','odd']:
        h = np.histogram(energies[patch], bins=n_bins, density=True)
        epsilon[patch] = 0.5 * (h[1][0:-1] + h[1][1:])
        rho[patch] = h[0]
        delta[patch] = h[1][1]-h[1][0]


    ### Build the local hamiltonian, separate into nodal and antinodal (inner and outer) spaces according to DCA procedure
    cn, cn_dag, nn = {}, {}, {}
    patch_num_map = {'even':1, 'odd':2}
    for spin in ['up','down']:
        cn['1-%s'%spin] = (c('even-%s'%spin,0) + c('odd-%s'%spin,0)) / np.sqrt(2)
        cn['2-%s'%spin] = (c('even-%s'%spin,0) - c('odd-%s'%spin,0)) / np.sqrt(2)
        nn['1-%s'%spin] = dagger(cn['1-%s'%spin]) * cn['1-%s'%spin]
        nn['2-%s'%spin] = dagger(cn['2-%s'%spin]) * cn['2-%s'%spin]
    h_loc = U * (nn['1-up'] * nn['1-down'] + nn['2-up'] * nn['2-down'])


    ### Actually this greens function is separated into 4 blocks, nice for us
    S = Solver(beta = beta, gf_struct = [('even-up',1), ('odd-up',1), ('even-down',1), ('odd-down',1)], n_l = 100)
    G = S.G0_iw.copy()



    ### Guess chemical potential as starting point for self-energy for now
    S.Sigma_iw << mu

    n_loops = 10
    n_length_cycles = 100
    n_cycles = int(5e4)
    true_n_cycles = mpi.size*n_cycles
    outname = f"vbhubbard_data/results_{mu:.3f}_{int(true_n_cycles/int(10**int(np.log10(true_n_cycles))))}e{int(np.log10(true_n_cycles))}_{n_length_cycles}.h5"
    
    old_siw = 0
    dens_matrix = {}
    conv_values = {}
    for nloop in range(n_loops):

        ### Bit messy, but basically just integrating over DOS while fixing self energy (since its momentum independent)
        G.zero()
        for spin in ['up', 'down']:
            for patch in ['even', 'odd']:
                for i in range(n_bins):
                    G['%s-%s'%(patch,spin)] += rho[patch][i] * delta[patch] * \
                        inverse(iOmega_n + mu - epsilon[patch][i] - S.Sigma_iw['%s-%s'%(patch,spin)])
        
        ### Once we find lattice greens, put it back into initial guess for Weiss field
        for block, g0 in S.G0_iw:
            g0 << inverse(inverse(G[block]) + S.Sigma_iw[block])
        
        # if nloop == n_loops-1:
        #     n_cycles = n_cycles*10
        Sigma_old = S.Sigma_iw.copy()
        S.solve(h_int = h_loc,                           
                n_cycles  = n_cycles,                       
                length_cycle = n_length_cycles,                       
                n_warmup_cycles = 10000,                 
                measure_G_l = False,  ### Lets not measure Gl for now     
                measure_density_matrix=True,
                use_norm_as_weight=True,
                random_name='mt11213b',
                random_seed=mpi.rank + np.random.randint(1, 10000)*nloop + int(1000*mu)
                )                 

        ### Save original for comparison
        if mpi.is_master_node():
            with HDFArchive(outname) as A:
                A['G_iw_org-%i'%nloop] = S.G_iw
                A['G_tau_org-%i'%nloop] = S.G_tau
                A['Sigma_iw_org-%i'%nloop] = S.Sigma_iw
                A['G0_iw-%i'%nloop] = S.G0_iw

            ### Here we recalculate the new self-energy according to denoised impurity Greens
            
            Aws = {}
            for spin in ['up', 'down']:
                for patch in ['even', 'odd']:
                    name = f'{patch}-{spin}'
                    plot = False
                    # if spin == 'up' and nloop > 3:
                    #     plot = True
                    
                    n_fit_min = 200
                    density = trace_rho_op(S.density_matrix, n(name,0), S.h_loc_diagonalization)
                    dens_matrix[name] = density
                    gtau, g, aw = HD.gen_clean_giw(dens_matrix[name], S.G_iw[name], S.G_tau[name], S.G_moments[name],\
                                            S.Sigma_iw[name], S.G0_iw[name], S.Sigma_moments[name], n_fit_min)

                    Sig_calc = inverse(S.G0_iw[name]) - inverse(g)

        
                    ### Just take new self energy as solution
                    Sigma_iw_meas = S.Sigma_iw[name].copy()
                    S.Sigma_iw[name] << Sig_calc
                    S.G_iw[name] << g 
                    S.G_tau[name] << gtau
                    Aws[name] = aw

                    conv = calc_conv(Sigma_old[name], S.Sigma_iw[name])
                    conv_values[name] = conv
                    print(nloop, name, " Convergence:", conv, "Density:", dens_matrix[name])

                
            with HDFArchive(outname) as A:
                A['G_iw-%i'%nloop] = S.G_iw
                A['G_tau-%i'%nloop] = S.G_tau
                A['Sigma_iw-%i'%nloop] = S.Sigma_iw
                A['Densities-%i'%nloop] = dens_matrix
                A['Convergence-%i'%nloop] = conv_values
                A['Aws-%i'%nloop] = Aws
        # exit()
        mpi.barrier()
        sigma_new = mpi.bcast(S.Sigma_iw if mpi.is_master_node() else None)
        giw_new = mpi.bcast(S.G_iw if mpi.is_master_node() else None)
        gtau_new = mpi.bcast(S.G_tau if mpi.is_master_node() else None)

        for spin in ['up', 'down']:
            for patch in ['even', 'odd']:
                name = f'{patch}-{spin}'
                S.Sigma_iw[name] << sigma_new[name]
                S.G_iw[name] << giw_new[name]
                S.G_tau[name] << gtau_new[name]



