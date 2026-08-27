from pathlib import Path
import torch
from gtaudenoise.utilities.utils import *

Path("Data").mkdir(parents=True, exist_ok=True)
beta = 200.0
N_tau = 2**10 + 1
N_ws = 1000
ws_max = 10
N = 8 * 1024
test_fraction = 0.0625
training_testing_points = int(N * (1.0 + test_fraction))
set_beta(beta, N_tau)
set_ws(N_ws, ws_max)
giws, gtaus, Aws = sample_Aw(training_testing_points, num_gauss=24, wr_cutoff=8)
torch.save({"gtaus_train": gtaus[:N], "giws_train": giws[:N], "Aws_train": Aws[:N], "gtaus_test": gtaus[N:], "giws_test": giws[N:], "Aws_test": Aws[N:], "beta": beta, "ws_max": ws_max, "N_tau": N_tau, "N_ws": N_ws, "num_gauss": 24, "wr_cutoff": 8}, "Data/train_test_data.pt")
