#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################################################################
# simulation of catalog continuation (for forecasting)
#
# as described by Mizrahi et al., 2021
# Leila Mizrahi, Shyam Nandan, Stefan Wiemer;
# Embracing Data Incompleteness for Better Earthquake Forecasting.
# Journal of Geophysical Research: Solid Earth.
# doi: https://doi.org/10.1029/2021JB022379
##############################################################################


import pandas as pd
import numpy as np
import datetime as dt
from numpy import array
import json
import geopandas as gpd
from shapely.geometry import Polygon
import pprint

from simulation import simulate_catalog_continuation
from inversion import parameter_dict2array, parameter_array2dict, round_half_up, responsibility_factor

if __name__ == '__main__':

	forecast_duration = 365  # in days

	data_path = 'data/'

	fn_store_simulation = data_path + 'my_catalog_continuation.csv'
	fn_parameters = data_path + 'parameters.json'
	fn_evol = data_path + 'parameter_evolution.csv'
	fn_ip = data_path + 'background_probabilities.csv'
	fn_src = data_path + 'sources.csv'
	fn_data = data_path + 'simulated_continuation.csv'

	# read parameters
	with open(fn_parameters, 'r') as f:
		parameters_dict = json.load(f)

	aux_start = pd.to_datetime(parameters_dict["auxiliary_start"])
	forecast_start_date = pd.to_datetime(parameters_dict["timewindow_end"])
	forecast_end_date = forecast_start_date + dt.timedelta(days=int(forecast_duration))
	coordinates = np.array(
		[np.array(a) for a in eval(parameters_dict["shape_coords"])]
	)
	poly = Polygon(coordinates)

	fn_train_catalog = parameters_dict["fn"]
	delta_m = parameters_dict["delta_m"]
	m_ref = (parameters_dict["m_ref"])
	beta = parameters_dict["beta"]
	t_R = parameters_dict["t_R"]

	# read in correct ETAS parameters to be used for simulation
	evol = pd.read_csv(fn_evol, index_col=0)
	parameters = parameter_array2dict(evol.iloc[-1, :-2])
	theta = parameter_dict2array(parameters)
	theta_without_mu = theta[1:]
	print("using parameters calculated on", parameters_dict["calculation_date"], "\n")
	pprint.pprint(parameters)

	# read training catalog and source info (contains current rate needed for inflation factor calculation)
	catalog = pd.read_csv(fn_train_catalog, index_col=0, parse_dates=["time"], dtype={"url": str, "alert": str})
	sources = pd.read_csv(fn_src, index_col=0)
	# xi_plus_1 is aftershock productivity inflation factor
	sources["xi_plus_1"] = responsibility_factor(theta, beta, sources["source_current_rate"], t_R)

	catalog = pd.merge(
		sources,
		catalog[["latitude", "longitude", "time", "magnitude"]],
		left_index=True,
		right_index=True,
		how='left',
	)
	assert len(catalog) == len(sources), "lost/found some sources in the merge! " + str(len(catalog)) + " -- " + str(
		len(sources))
	assert catalog.magnitude.min() == m_ref, "smallest magnitude in sources is " + str(
		catalog.magnitude.min()) + " but I am supposed to simulate above " + str(m_ref)

	# background rates
	ip = pd.read_csv(fn_ip, index_col=0)
	ip.query("magnitude>=@m_ref -@delta_m/2", inplace=True)
	ip = gpd.GeoDataFrame(ip, geometry=gpd.points_from_xy(ip.latitude, ip.longitude))
	ip = ip[ip.intersects(poly)]

	# other constants
	coppersmith_multiplier = parameters_dict["coppersmith_multiplier"]
	earth_radius = parameters_dict["earth_radius"]

	print("m ref:", m_ref, "min magnitude in training catalog:", catalog["magnitude"].min())

	start = dt.datetime.now()

	continuation = simulate_catalog_continuation(
		catalog,
		auxiliary_start=aux_start,
		auxiliary_end=forecast_start_date,
		polygon=poly,
		simulation_end=forecast_end_date,
		parameters=parameters,
		mc=m_ref - delta_m / 2,
		beta_main=beta,
		verbose=False,
		background_lats=ip["latitude"],
		background_lons=ip["longitude"],
		background_probs=ip["P_background"],
		gaussian_scale=0.1
	)
	continuation.query(
		"time>=@forecast_start_date and time<=@forecast_end_date and magnitude >= @m_ref-@delta_m/2",
		inplace=True
	)

	print("took", dt.datetime.now() - start, "to simulate 1 catalog containing", len(continuation), "events.")

	continuation.magnitude = round_half_up(continuation.magnitude, 1)
	continuation.index.name = 'id'
	print("store catalog..")
	continuation[["latitude", "longitude", "time", "magnitude"]].to_csv(fn_store_simulation)
	print("\nDONE!")
