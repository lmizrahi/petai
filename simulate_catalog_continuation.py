import numpy as np
import datetime as dt
from shapely.geometry import Polygon


from simulation import generate_catalog
from inversion import round_half_up

import sys

from simulate_catalogs import simulate_catalog_continuation, transform_parameters
from inversion import parameter_dict2array
from earth_surface import round_half_up
from gr_est import estimate_beta_tinti
from etas_em_mc_var import filter_magnitudes_var_mc


import pandas as pd
import numpy as np
import datetime as dt
from pandas import Timestamp
from numpy import array
import datetime
import json
import geopandas as gpd
from shapely.geometry import Polygon
import os
import scipy.io as sio

period = 7
timewindow_end = dt.datetime(2020, 1, 1)

data_path = 'data/'
store_path = '/Volumes/LaCie/prdi/simulations/'
n_simulations_overall = 1000

n_simulations = n_simulations_overall // tot_files
if file_no == tot_files - 1:
	n_simulations += n_simulations_overall % tot_files


fn_parameters = data_path + 'parameters.json'
fn_evol = data_path + str(timewindow_end)[:10] + '_evol_mc_31.csv'


fn_ip = data_path + 'ip_' + str(timewindow_end)[:10] + '_' + mode + '.csv'
fn_src = data_path + 'src_' + str(timewindow_end)[:10] + '_' + mode + '.csv'


fn_data = store_path + 'sim_' + str(timewindow_end)[:10] + '_' + mode \
		  + '_' + str(period) + 'D_file_' + str(file_no) + '.h5'


omori_tau = True


with open(fn_parameters, 'r') as f:
	parameters_dict = json.load(f)
parameters = eval(parameters_dict["final_parameters"])
aux_start = pd.to_datetime(parameters_dict["auxiliary_start"])
tw_start = pd.to_datetime(parameters_dict["timewindow_start"])
train_end = pd.to_datetime(parameters_dict["timewindow_end"])
tw_end = train_end + dt.timedelta(days=int(period))
coordinates = np.array(
	[np.array(a) for a in eval(parameters_dict["shape_coords"])]
)

fn = parameters_dict["fn"]
delta_m = parameters_dict["delta_m"]
mc_vals = eval(parameters_dict["mc_vals"])
mc_times = np.array([pd.to_datetime(a) for a in eval(parameters_dict["mc_times"])])
mc_ref = parameters_dict["mc_ref"]
beta = parameters_dict["beta"]
mc_min = mc_ref

catalog = pd.read_csv(fn, index_col=0, parse_dates=["time"], dtype={"url": str, "alert": str})
sources = pd.read_csv(fn_src, index_col=0)

catalog = pd.merge(
	sources,
	catalog[["latitude", "longitude", "time", "magnitude"]],
	left_index=True,
	right_index=True,
	how='left',
)
assert len(catalog) == len(sources), "lost/found some sources in the merge! " + str(len(catalog)) + " -- " + str(len(sources))

poly = Polygon(coordinates)


# background rates
ip = pd.read_csv(fn_ip, index_col=0)
ip.query("magnitude>=@mc_min -@delta_m/2", inplace=True)
ip = gpd.GeoDataFrame(ip, geometry=gpd.points_from_xy(ip.longitude, ip.latitude))
ip = ip[ip.intersects(poly)]


# other constants
coppersmith_multiplier = parameters_dict["coppersmith_multiplier"]
earth_radius = parameters_dict["earth_radius"]

print("using parameters calculated on", parameters_dict["calculation_date"], "\n")


theta = parameter_dict2array(parameters)
theta_without_mu = theta[1:]






print("mc ref:", mc_ref, "min magnitude in training catalog:", catalog["magnitude"].min())

np.random.seed(777)
start = dt.datetime.now()

continuation = simulate_catalog_continuation(
	catalog,
	auxiliary_start=aux_start,
	auxiliary_end=train_end,
	polygon=poly,
	simulation_end=tw_end,
	parameters=parameters,
	mc=mc_ref - delta_m / 2,
	beta_main=beta,
	omori_tau=omori_tau,
	verbose=False,
	# mc_min=mc_min-delta_m/2,
	background_lats=ip["latitude"],
	background_lons=ip["longitude"],
	background_probs=ip["P_background"],
	gaussian_scale=0.1
)
relevant = continuation.query("time>=@train_end and time<=@tw_end").copy()

print("took", dt.datetime.now() - start)




if __name__ == '__main__':
	fn_store = 'my_synthetic_catalog.csv'
	shape_coords = np.load("california_shape.npy")
	caliregion = Polygon(shape_coords)
	burn_start = dt.datetime(1871, 1, 1)
	primary_start = dt.datetime(1971, 1, 1)
	end = dt.datetime(2021, 1, 1)

	delta_m = 0.1
	mc = 3.6
	beta = np.log(10)

	parameters = {
		'log10_mu': -7.5,
		'log10_k0': -2.49,
		'a': 1.69,
		'log10_c': -2.95,
		'omega': -0.03,
		'log10_tau': 3.99,
		'log10_d': -0.35,
		'gamma': 1.22,
		'rho': 0.51
	}

	# np.random.seed(777)

	synthetic = generate_catalog(
		polygon=caliregion,
		timewindow_start=burn_start,
		timewindow_end=end,
		parameters=parameters,
		mc=mc,
		beta_main=beta,
		delta_m=delta_m
	)

	synthetic.magnitude = round_half_up(synthetic.magnitude, 1)
	synthetic.index.name = 'id'
	print("store catalog..")
	synthetic[["latitude", "longitude", "time", "magnitude"]].query("time>=@primary_start").to_csv(fn_store)
	print("\nDONE!")