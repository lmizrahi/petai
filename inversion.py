#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################################################################
# functions needed for inversion of ETAS and detection parameters (for PETAI)
#
# as described by Mizrahi et al., 2021
# Leila Mizrahi, Shyam Nandan, Stefan Wiemer;
# Embracing Data Incompleteness for Better Earthquake Forecasting.
# Journal of Geophysical Research: Solid Earth.
# doi: https://doi.org/10.1029/2021JB022379
##############################################################################

import pandas as pd
import datetime as dt
import numpy as np
import geopandas as gpd

import os
import sys
import json
import pprint
from textwrap import indent

import pyproj
from shapely.geometry import Polygon
import shapely.ops as ops
from functools import partial

from scipy.optimize import minimize
from scipy.special import gamma as gamma_func, gammaln, gammaincc, exp1, beta as beta_func


def round_half_up(n, decimals=0):
	multiplier = 10 ** decimals
	return np.floor(n*multiplier + 0.5) / multiplier


def estimate_beta_tinti(magnitudes, mc, weights=None, axis=None, delta_m=0):
	if delta_m > 0:
		p = (1 + (delta_m / (np.average(magnitudes - mc, weights=weights, axis=axis))))
		beta = 1 / delta_m * np.log(p)
	else:
		beta = 1 / np.average((magnitudes - (mc - delta_m/2)), weights=weights, axis=axis)
	return beta


def coppersmith(mag, typ):
	# result is in km

	# typ is one of the following:
	# 1: strike slip fault
	# 2: reverse fault
	# 3: normal fault
	# 4: oblique fault

	if typ == 1:
		# surface rupture length
		SRL = np.power(10, (0.74 * mag - 3.55))
		# subsurface rupture length
		SSRL = np.power(10, (0.62 * mag - 2.57))
		# rupture width
		RW = np.power(10, (0.27 * mag - 0.76))
		# rupture area
		RA = np.power(10, (0.9 * mag - 3.42))
		# average slip
		AD = np.power(10, (0.9 * mag - 6.32))

	elif typ == 2:
		# surface rupture length
		SRL = np.power(10, (0.63 * mag - 2.86))
		# subsurface rupture length
		SSRL = np.power(10, (0.58 * mag - 2.42))
		# rupture width
		RW = np.power(10, (0.41 * mag - 1.61))
		# rupture area
		RA = np.power(10, (0.98 * mag - 3.99))
		# average slip
		AD = np.power(10, (0.08 * mag - 0.74))

	elif typ == 3:
		# surface rupture length
		SRL = np.power(10, (0.5 * mag - 2.01))
		# subsurface rupture length
		SSRL = np.power(10, (0.5 * mag - 1.88))
		# rupture width
		RW = np.power(10, (0.35 * mag - 1.14))
		# rupture area
		RA = np.power(10, (0.82 * mag - 2.87))
		# average slip
		AD = np.power(10, (0.63 * mag - 4.45))

	elif typ == 4:
		# surface rupture length
		SRL = np.power(10, (0.69 * mag - 3.22))
		# subsurface rupture length
		SSRL = np.power(10, (0.59 * mag - 2.44))
		# rupture width
		RW = np.power(10, (0.32 * mag - 1.01))
		# rupture area
		RA = np.power(10, (0.91 * mag - 3.49))
		# average slip
		AD = np.power(10, (0.69 * mag - 4.80))

	return {
		'SRL': SRL,
		'SSRL': SSRL,
		'RW': RW,
		'RA': RA,
		'AD': AD
	}


def to_days(timediff):
	return timediff / dt.timedelta(days=1)


def upper_gamma_ext(a, x):
	if a > 0:
		return gammaincc(a, x) * gamma_func(a)
	elif a == 0:
		return exp1(x)
	else:
		return (upper_gamma_ext(a + 1, x) - np.power(x, a)*np.exp(-x)) / a


def hav(theta):
	return np.square(np.sin(theta / 2))


def haversine(lat_rad_1, lat_rad_2, lon_rad_1, lon_rad_2, earth_radius=6.3781e3):
	# lat_rad_1, lat_rad_2, lon_rad_1, lon_rad_2 = latlons

	# to calculate distance on a sphere
	d = 2 * earth_radius * np.arcsin(
		np.sqrt(
			hav(lat_rad_1 - lat_rad_2)
			+ np.cos(lat_rad_1)
			* np.cos(lat_rad_2)
			* hav(lon_rad_1 - lon_rad_2)
		)
	)
	return d


def polygon_surface(polygon):
	geom_area = ops.transform(
		partial(
			pyproj.transform,
			pyproj.Proj('EPSG:4326'),
			pyproj.Proj(
				proj='aea',
				lat_1=polygon.bounds[0],
				lat_2=polygon.bounds[2])),
		polygon)
	return geom_area.area / 1e6


def parameter_array2dict(theta):
	return dict(zip(
		['log10_mu', 'log10_k0', 'a', 'log10_c', 'omega', 'log10_tau', 'log10_d', 'gamma', 'rho'],
		theta
	))


def parameter_dict2array(parameters):
	order = ['log10_mu', 'log10_k0', 'a', 'log10_c', 'omega', 'log10_tau', 'log10_d', 'gamma', 'rho']
	return np.array([
		parameters[key] for key in order
	])


def branching_ratio_tapered(theta, beta):
	log10_mu, log10_k0, a, log10_c, omega, log10_tau, log10_d, gamma, rho = theta
	k0 = np.power(10, log10_k0)
	c = np.power(10, log10_c)
	d = np.power(10, log10_d)
	tau = np.power(10, log10_tau)

	eta = beta * k0 * np.pi * np.power(d, -rho) * np.power(tau, -omega) * np.exp(c / tau) * upper_gamma_ext(-omega,
																											c / tau) / (
				  rho * (-a + beta + gamma * rho))
	return eta


def transform_parameters(par, beta, delta_m):
	par_corrected = par.copy()

	par_corrected["log10_mu"] -= delta_m * beta / np.log(10)
	par_corrected["log10_d"] += delta_m * par_corrected["gamma"] / np.log(10)
	par_corrected["log10_k0"] += delta_m * par_corrected["gamma"] * par_corrected["rho"] / np.log(10)

	return par_corrected


def expected_aftershocks(event, params, no_start=False, no_end=False):
	theta, mc = params

	log10_k0, a, log10_c, omega, log10_tau, log10_d, gamma, rho = theta
	k0 = np.power(10, log10_k0)
	c = np.power(10, log10_c)
	tau = np.power(10, log10_tau)
	d = np.power(10, log10_d)

	if no_start:
		if no_end:
			event_magnitude = event
		else:
			event_magnitude, event_time_to_end = event
	else:
		if no_end:
			event_magnitude, event_time_to_start = event
		else:
			event_magnitude, event_time_to_start, event_time_to_end = event

	number_factor = k0 * np.exp(a * (event_magnitude - mc))
	area_factor = np.pi * np.power(
		d * np.exp(gamma * (event_magnitude - mc)),
		-1 * rho
	) / rho

	time_factor = np.exp(c/tau) * np.power(tau, -omega)  # * gamma_func(-omega)

	if no_start:
		time_fraction = upper_gamma_ext(-omega, c/tau)
	else:
		time_fraction = upper_gamma_ext(-omega, (event_time_to_start + c)/tau)
	if not no_end:
		time_fraction = time_fraction - upper_gamma_ext(-omega, (event_time_to_end + c)/tau)

	time_factor = time_factor * time_fraction

	return number_factor * area_factor * time_factor


def neg_log_likelihood(theta, args):
	n_hat, Pij, source_events, timewindow_length, timewindow_start, area, beta, mc_min = args

	assert Pij.index.names == ("source_id", "target_id"), "Pij must have multiindex with names 'source_id', 'target_id'"
	assert source_events.index.name == "source_id", "source_events must have index with name 'source_id'"

	log10_k0, a, log10_c, omega, log10_tau, log10_d, gamma, rho = theta
	k0 = np.power(10, log10_k0)
	c = np.power(10, log10_c)
	tau = np.power(10, log10_tau)
	d = np.power(10, log10_d)

	source_events["G"] = expected_aftershocks(
		[
			source_events["source_magnitude"],
			source_events["pos_source_to_start_time_distance"],
			source_events["source_to_end_time_distance"]
		],
		[theta, mc_min]
	)

	aftershock_term = ll_aftershock_term(
		source_events["l_hat"],
		source_events["G"],
	).sum()

	# space time distribution term
	Pij["likelihood_term"] = (
			(omega * np.log(tau) - np.log(upper_gamma_ext(-omega, c/tau))
			 + np.log(rho) + rho * np.log(
						d * np.exp(gamma * (Pij["source_magnitude"] - mc_min))
					))
			- ((1 + rho) * np.log(
		Pij["spatial_distance_squared"] + (
				d * np.exp(gamma * (Pij["source_magnitude"] - mc_min))
		)
	))
			- (1 + omega) * np.log(Pij["time_distance"] + c)
			- (Pij["time_distance"] + c) / tau
			- np.log(np.pi)

	)
	distribution_term = Pij["Pij"].mul(Pij["likelihood_term"]).sum()

	total = aftershock_term + distribution_term

	return -1 * total


def optimize_parameters(theta_0, ranges, args, verbose=False):
	start_calc = dt.datetime.now()

	n_hat, Pij, source_events, timewindow_length, timewindow_start, area, beta, mc_min = args
	log10_mu_range, log10_k0_range, a_range, log10_c_range, omega_range, log10_tau_range, log10_d_range, gamma_range, rho_range = ranges

	log10_mu, log10_k0, a, log10_c, omega, log10_tau, log10_d, gamma, rho = theta_0
	mu = np.power(10, log10_mu)
	k0 = np.power(10, log10_k0)
	c = np.power(10, log10_c)
	tau = np.power(10, log10_tau)
	d = np.power(10, log10_d)

	# estimate mu independently and remove from parameters
	mu_hat = n_hat / (area * timewindow_length)
	theta_0_without_mu = log10_k0, a, log10_c, omega, log10_tau, log10_d, gamma, rho

	bounds = [
		log10_k0_range,
		a_range,
		log10_c_range,
		omega_range,
		log10_tau_range,
		log10_d_range,
		gamma_range,
		rho_range
	]

	res = minimize(
		neg_log_likelihood,
		x0=theta_0_without_mu,
		bounds=bounds,
		args=args,
		tol=1e-12,
	)

	# print(res)
	new_theta_without_mu = res.x
	new_theta = [np.log10(mu_hat), *new_theta_without_mu]

	if verbose:
		print("optimization step took ", dt.datetime.now() - start_calc)

	return np.array(new_theta)


def prepare_catalog(
		data,
		coppersmith_multiplier,
		timewindow_start,
		timewindow_end,
		earth_radius,
		delta_m=0
):

	calc_start = dt.datetime.now()
	# precalculates distances in time and space between events that are potentially relate to each other

	# only use data above completeness magnitude
	if delta_m > 0:
		data["magnitude"] = round_half_up(data["magnitude"] / delta_m) * delta_m

	data.sort_values(by='time', inplace=True)

	# all entries can be sources, but targets only after timewindow starts
	targets = data.query("time>=@timewindow_start").copy()

	# calculate some source stuff
	data["distance_range_squared"] = np.square(
		coppersmith(data["magnitude"], 4)["SSRL"] * coppersmith_multiplier
	)
	# time distances for all mc windows
	data["source_to_end_time_distance"] = to_days(timewindow_end - data["time"])
	data["pos_source_to_start_time_distance"] = np.clip(
		to_days(timewindow_start - data["time"]),
		a_min=0,
		a_max=None
	)

	# translate target lat, lon to radians for spherical distance calculation
	targets['target_lat_rad'] = np.radians(targets['latitude'])
	targets['target_lon_rad'] = np.radians(targets['longitude'])
	targets["target_time"] = targets["time"]
	targets["target_id"] = targets.index
	targets["target_time"] = targets["time"]
	targets["target_current_rate"] = targets["current_rate"]
	# columns that are needed later
	targets["source_id"] = 'i'
	targets["source_magnitude"] = 0.0
	targets["source_current_rate"] = 0.0
	targets["time_distance"] = 0.0
	targets["spatial_distance_squared"] = 0.0
	targets["source_to_end_time_distance"] = 0.0
	targets["pos_source_to_start_time_distance"] = 0.0

	targets = targets.sort_values(by="time")
	targets_all = targets.copy()

	# define index and columns that are later going to be needed
	if pd.__version__ >= '0.24.0':
		index = pd.MultiIndex(
			levels=[[], []],
			names=["source_id", "target_id"],
			codes=[[], []]
		)
	else:
		index = pd.MultiIndex(
			levels=[[], []],
			names=["source_id", "target_id"],
			labels=[[], []]
		)
	columns = [
		"target_time",
		"source_magnitude",
		"source_current_rate",
		"target_current_rate",
		"spatial_distance_squared",
		"time_distance",
		"source_to_end_time_distance",
		"pos_source_to_start_time_distance",
	]
	res_df = pd.DataFrame(index=index, columns=columns)

	df_list = []

	print('  number of sources:', len(data.index), '\n')
	print('  number of targets:', len(targets.index), '\n')
	for source in data.itertuples():
		stime = source.time
		# if verbose:# and np.random.uniform() < 0.1:
		#    print(stime)

		# filter potential targets
		if source.time < timewindow_start:
			potential_targets = targets.copy()
		else:
			potential_targets = targets.query(
				"time>@stime"
			).copy()
		targets = potential_targets.copy()

		if potential_targets.shape[0] == 0:
			continue

		# get values of source event
		slatrad = np.radians(source.latitude)
		slonrad = np.radians(source.longitude)
		drs = source.distance_range_squared

		# get source id and info of target events
		potential_targets["source_id"] = source.Index
		potential_targets["source_magnitude"] = source.magnitude
		potential_targets["source_current_rate"] = source.current_rate

		# calculate space and time distance from source to target event
		potential_targets["time_distance"] = to_days(potential_targets["target_time"] - stime)

		potential_targets["spatial_distance_squared"] = np.square(
			haversine(
				slatrad,
				potential_targets['target_lat_rad'],
				slonrad,
				potential_targets['target_lon_rad'],
				earth_radius
			)
		)

		# filter for only small enough distances
		potential_targets.query("spatial_distance_squared <= @drs", inplace=True)

		# calculate time distance from source event to timewindow boundaries for integration later
		potential_targets["source_to_end_time_distance"] = source.source_to_end_time_distance
		potential_targets["pos_source_to_start_time_distance"] = source.pos_source_to_start_time_distance

		# append to resulting dataframe
		df_list.append(potential_targets)

	res_df = pd.concat(df_list)[["source_id", "target_id"] + columns].reset_index().set_index(
		["source_id", "target_id"])
	print('\n   took', (dt.datetime.now() - calc_start), 'to prepare the data\n')

	return res_df


def set_initial_values(ranges):

	log10_mu_range, log10_k0_range, a_range, log10_c_range, omega_range, log10_tau_range, log10_d_range, gamma_range, rho_range = ranges

	log10_mu = np.random.uniform(*log10_mu_range)
	log10_k0 = np.random.uniform(*log10_k0_range)
	a = np.random.uniform(*a_range)
	log10_c = np.random.uniform(*log10_c_range)
	omega = np.random.uniform(*omega_range)
	log10_tau = np.random.uniform(*log10_tau_range)
	log10_d = np.random.uniform(*log10_d_range)
	gamma = np.random.uniform(*gamma_range)
	rho = np.random.uniform(*rho_range)

	return [
		log10_mu,
		log10_k0,
		a,
		log10_c,
		omega,
		log10_tau,
		log10_d,
		gamma,
		rho
	]


def triggering_kernel(metrics, params):
	# given time distance in days and squared space distance in square km and magnitude of target event,
	# calculate the (not normalized) likelihood, that source event triggered target event

	time_distance, spatial_distance_squared, m = metrics
	theta, mc = params

	log10_mu, log10_k0, a, log10_c, omega, log10_tau, log10_d, gamma, rho = theta
	mu = np.power(10, log10_mu)
	k0 = np.power(10, log10_k0)
	c = np.power(10, log10_c)
	tau = np.power(10, log10_tau)
	d = np.power(10, log10_d)

	aftershock_number = k0 * np.exp(a * (m - mc))
	time_decay = np.exp(-time_distance / tau) / np.power((time_distance + c), (1 + omega))
	space_decay = 1 / np.power(
		(spatial_distance_squared + d * np.exp(gamma * (m - mc))),
		(1 + rho)
	)

	res = aftershock_number * time_decay * space_decay
	return res


def triggering_rate(metrics, params):
	# given time distance in days and magnitude of source event,
	# calculate the infinite region aftershock rate at that time

	time_distance, source_m = metrics
	theta, mc = params

	log10_mu, log10_k0, a, log10_c, omega, log10_tau, log10_d, gamma, rho = theta
	mu = np.power(10, log10_mu)
	k0 = np.power(10, log10_k0)
	c = np.power(10, log10_c)
	tau = np.power(10, log10_tau)
	d = np.power(10, log10_d)

	aftershock_number = k0 * np.exp((a - rho * gamma) * (source_m - mc))
	space_integral = np.pi / (rho * np.power(d, rho))

	time_decay = np.exp(-time_distance / tau) / np.power((time_distance + c), (1 + omega))

	res = aftershock_number * time_decay * space_integral
	return res


def responsibility_factor(theta, beta, rate, t_R):
	log10_mu, log10_k0, a, log10_c, omega, log10_tau, log10_d, gamma, rho = theta

	k = - (a - beta - gamma*rho) / beta

	xi_plus_1 = 1 / (
			k * beta_func(k, rate * t_R + 1)
	)

	return xi_plus_1


def observation_factor(rate, t_R):

	zeta_plus_1 = rate * t_R + 1

	return zeta_plus_1


def expectation_step(distances, target_events, source_events, params, verbose=False):
	calc_start = dt.datetime.now()
	theta, beta, t_R, mc_min = params
	log10_mu, log10_k0, a, log10_c, omega, log10_tau, log10_d, gamma, rho = theta
	# print('I am doing the expectation step with parameters', theta)
	mu = np.power(10, log10_mu)
	k0 = np.power(10, log10_k0)
	c = np.power(10, log10_c)
	tau = np.power(10, log10_tau)
	d = np.power(10, log10_d)

	# calculate the triggering density values gij
	if verbose:
		print('    calculating gij\n')
	Pij_0 = distances.copy()
	Pij_0["gij"] = triggering_kernel(
		[
			Pij_0["time_distance"],
			Pij_0["spatial_distance_squared"],
			Pij_0["source_magnitude"]
		],
		[theta, mc_min]
	)

	# responsibility factor for invisible triggering events
	Pij_0["xi_plus_1"] = responsibility_factor(theta, beta, Pij_0["source_current_rate"], t_R)
	Pij_0["zeta_plus_1"] = observation_factor(Pij_0["target_current_rate"], t_R)
	# calculate muj for each target. currently constant, could be improved
	target_events_0 = target_events.copy()
	target_events_0["mu"] = mu

	# calculate triggering probabilities Pij
	if verbose:
		print('    calculating Pij\n')
	Pij_0["tot_rates"] = 0
	Pij_0["tot_rates"] = Pij_0["tot_rates"].add((Pij_0["gij"] * Pij_0["xi_plus_1"]).sum(level=1)).add(target_events_0["mu"])
	Pij_0["Pij"] = Pij_0["gij"].div(Pij_0["tot_rates"])

	# calculate probabilities of being triggered or background
	target_events_0["P_triggered"] = 0
	target_events_0["P_triggered"] = target_events_0["P_triggered"].add(Pij_0["Pij"].sum(level=1)).fillna(0)
	target_events_0["P_background"] = target_events_0["mu"] / Pij_0.groupby(level=1).first()["tot_rates"]
	target_events_0["P_background"] = target_events_0["P_background"].fillna(1)
	target_events_0["zeta_plus_1"] = Pij_0.groupby(level=1).first()["zeta_plus_1"]
	target_events_0["zeta_plus_1"] = target_events_0["zeta_plus_1"].fillna(1)

	# calculate expected number of background events
	if verbose:
		print('    calculating n_hat\n')
	n_hat_0 = (target_events_0["P_background"] * target_events_0["zeta_plus_1"]).sum()

	# calculate aftershocks per source event
	source_events_0 = source_events.copy()
	source_events_0["l_hat"] = (Pij_0["Pij"] * Pij_0["zeta_plus_1"]).sum(level=0)
	if verbose:
		print('expectation step took ', dt.datetime.now() - calc_start)
	return Pij_0, target_events_0, source_events_0, n_hat_0


def ll_aftershock_term(l_hat, g):
	mask = g != 0
	term = -1 * gammaln(l_hat + 1) - g
	term = term + l_hat * np.where(mask, np.log(g, where=mask), -300)
	return term


def detection_neg_log_likelihood(theta, args):
	magnitudes, rates, m_0 = args
	beta, t_R = theta

	exponent = t_R * rates

	res = -(
			exponent * np.log(1 - np.exp(-beta * (magnitudes - m_0)))
			+ np.log(beta) - beta * (magnitudes - m_0) + np.log(exponent + 1)
	).sum()
	return res


def optimize_detection_parameters(theta_0, ranges, args, verbose=False, uncertainties=False):
	start_calc = dt.datetime.now()

	beta_range, t_R_range = ranges

	bounds = [
		beta_range,
		t_R_range
	]

	res = minimize(
		detection_neg_log_likelihood,
		x0=theta_0,
		bounds=bounds,
		args=args,
		tol=1e-12,
	)

	new_theta = res.x

	if verbose:
		print("optimization step took ", dt.datetime.now() - start_calc)
	if uncertainties:
		return np.array(new_theta), res
	return np.array(new_theta)


def estimate_rates(distances, theta, beta, t_R, mc, delta_m, verbose=False):
	log10_mu, log10_k0, a, log10_c, omega, log10_tau, log10_d, gamma, rho = theta
	mu = 10 ** log10_mu
	dists_needed = distances[["source_current_rate", "time_distance", "source_magnitude"]]

	estimated_aftershock_rates = (
			triggering_rate(
				[
					dists_needed["time_distance"],
					dists_needed["source_magnitude"]
				],
				[theta, mc - delta_m / 2]
			) * responsibility_factor(
		theta, beta, distances["source_current_rate"], t_R
	)
	).groupby("target_id").sum()

	diff = 100
	i = 0

	while diff > 1:
		dists_needed = pd.merge(
			dists_needed.drop("source_current_rate", axis=1),
			pd.DataFrame(estimated_aftershock_rates, columns=['source_current_rate']),
			left_on='source_id',
			right_index=True,
			how='left'
		)
		dists_needed.fillna(0, inplace=True)
		dists_needed["source_current_rate"] = dists_needed["source_current_rate"].add(mu)

		new_estimated_aftershock_rates = (
				triggering_rate(
					[
						dists_needed["time_distance"],
						dists_needed["source_magnitude"]
					],
					[theta, mc - delta_m / 2]
				) * responsibility_factor(
			theta, beta, dists_needed["source_current_rate"], t_R
		)
		).groupby("target_id").sum()

		diff = (np.abs(np.log(estimated_aftershock_rates / new_estimated_aftershock_rates)).sum())
		estimated_aftershock_rates = new_estimated_aftershock_rates
		i += 1
	distances = pd.merge(
		pd.merge(
			distances.drop(["source_current_rate", "target_current_rate"], axis=1),
			pd.DataFrame(estimated_aftershock_rates, columns=['source_current_rate']),
			left_on='source_id',
			right_index=True,
			how='left'
		),
		pd.DataFrame(estimated_aftershock_rates, columns=['target_current_rate']),
		left_on='target_id',
		right_index=True,
		how='left'
	)

	distances.fillna(0, inplace=True)
	distances["source_current_rate"] = distances["source_current_rate"].add(mu)
	distances["target_current_rate"] = distances["target_current_rate"].add(mu)

	if verbose:
		print("   rates converged after", i, "iterations")
	return distances, new_estimated_aftershock_rates


def rates_and_detection(targets, distances, theta, mc, delta_m, start_date, p_0=None, verbose=False, space="   "):
	p_0 = p_0 or [3.0, 1.0]
	ranges = [
		(0.4 * np.log(10), 3 * np.log(10)),
		(1e-7, 1e2)
	]
	log10_mu, log10_k0, a, log10_c, omega, log10_tau, log10_d, gamma, rho = theta
	mu = 10 ** log10_mu
	rates = targets.query("time>=@start_date")["current_rate"] * 0
	rates = rates.add(mu)
	diff = 1
	i = 0
	while diff > 1e-12:
		if verbose:
			print("optimizing detection parameters...")
		p_new = optimize_detection_parameters(
			p_0,
			ranges,
			[
				targets.query("time>=@start_date")["magnitude"],
				rates,
				mc - delta_m / 2
			]
		)
		beta_new, t_R_new = p_new
		if verbose:
			print("   new parameters:", p_new)
			print("optimizing rates ...")
		distances, aftershock_rates = estimate_rates(distances, theta, beta_new, t_R_new, mc, delta_m, verbose=verbose)
		rates = pd.Series(aftershock_rates, index=targets.index)
		rates.fillna(0, inplace=True)
		rates = rates.add(mu)

		diff = np.abs(p_0 - p_new).sum()
		p_0 = p_new
		i += 1

	targets["current_rate"] = rates
	targets["current_rate"].fillna(0, inplace=True)
	print(2*space, "converged after", i, "iterations")
	return p_new, targets, distances


def invert_etas_params(
		df,
		beta,
		t_R,
		mc_ref,
		delta_m,
		timewindow_length, timewindow_start, area,
		theta_0=None,
		distances=None,
		ranges=None,
		verbose=False,
		space=" "
):

	print(2*space, "beta is " + str(beta), " and t_R is ", t_R, '\n')


	###########
	# initialize
	###########

	print(2*space, 'initializing..\n')

	print(3*space, 'preparing source and target events\n')

	target_events = df.query("time > @ timewindow_start").copy()
	target_events.index.name = "target_id"

	source_columns = [
		"source_magnitude",
		"source_current_rate",
		"source_to_end_time_distance",
		"pos_source_to_start_time_distance"
	]

	source_events = pd.DataFrame(
		distances[
			source_columns
		].groupby("source_id").first()
	)

	if theta_0 is None:
		print(3*space, 'randomly chosing initial values for theta\n')
		initial_values = set_initial_values(ranges)
	else:
		initial_values = theta_0

	start_time = dt.datetime.now()

	for i in np.arange(0, 40, 1):
		iteration_start = dt.datetime.now()
		print(2*space, 'ETAS iteration ' + str(i) + '\n')

		if i == 0:
			parameters = initial_values

		print(3*space, 'expectation\n')
		Pij, target_events, source_events, n_hat = expectation_step(
			distances=distances,
			target_events=target_events,
			source_events=source_events,
			# params=[parameters, mc_ref - delta_m/2],
			params=[parameters, beta, t_R, mc_ref - delta_m / 2],
			verbose=verbose
		)
		print(4*space, 'n_hat:', n_hat, '\n')
		print(4*space, 'Pij shape:', Pij.shape, '\n')

		print(3*space, 'maximization\n')
		args = [n_hat, Pij, source_events, timewindow_length, timewindow_start, area, beta, mc_ref - delta_m / 2]

		new_parameters = optimize_parameters(
			theta_0=parameters,
			args=args,
			ranges=ranges
		)
		if verbose:
			print(4*space, 'new parameters:\n')
			print(indent(pprint.pformat(
				parameter_array2dict(new_parameters)
			), 4*space))
			print('\n')
		diff_to_before = np.sum(np.abs(parameters - new_parameters))
		print(4*space, 'difference to previous:', diff_to_before, '\n')

		br = branching_ratio_tapered(parameters, beta)
		print(4*space, 'branching ratio:', br, '\n')
		parameters = new_parameters

		print(4 * space, 'iteration', i, 'took', dt.datetime.now() - iteration_start, '\n')

		if diff_to_before < 0.001:
			print(2*space, 'stopping here.\n')
			break

	print(2*space, 'last expectation step\n')
	Pij, target_events, source_events, n_hat = expectation_step(
		distances=distances,
		target_events=target_events,
		source_events=source_events,
		params=[parameters, beta, t_R, mc_ref - delta_m / 2],
		verbose=verbose
	)
	print(3*space, 'n_hat:', n_hat, '\n')

	print(space, 'ETAS inversion took', dt.datetime.now() - start_time, '\n')
	return new_parameters, source_events, target_events, Pij


def invert_etas_and_detecetion_params(metadata, distances=None, globe=False, timewindow_end=None):

	start_time = dt.datetime.now()
	space = "   "
	print("INVERSION OF ETAS AND DETECTION PARAMETERS\n\n")
	if isinstance(metadata, str):
		with open(metadata, 'r') as f:
			parameters_dict = json.load(f)
	else:
		parameters_dict = metadata

	# reading metadata and data, catalog, etc...
	print('Preparing metadata...\n')
	fn_catalog = parameters_dict["fn_catalog"]
	print(space, "Catalog:", fn_catalog)

	if sys.platform == 'linux':
		data_path = parameters_dict["data_path_euler"]
	else:
		data_path = parameters_dict["data_path"]

	auxiliary_start = pd.to_datetime(parameters_dict["auxiliary_start"])
	timewindow_start = pd.to_datetime(parameters_dict["timewindow_start"])
	timewindow_end = timewindow_end or pd.to_datetime(parameters_dict["timewindow_end"])
	print(
		space, "Time Window: " + str(auxiliary_start)
			   + " (aux) - " + str(timewindow_start) + " (start) - " + str(timewindow_end) + " (end)"
	)

	m_ref = parameters_dict["m_ref"]
	delta_m = parameters_dict["delta_m"]
	print(space, "m_ref is " + str(m_ref) + " and delta_m is " + str(delta_m))

	try:
		mc = parameters_dict["mc"]
	except KeyError:
		mc = m_ref
	if distances is None:
		print(space, "discrete mc used in iteration 0 is " + str(mc))

	try:
		store_pij = parameters_dict["store_pij"]
	except KeyError:
		store_pij = False

	# defining some constants here..

	coppersmith_multiplier = parameters_dict["coppersmith_multiplier"]
	print(space, "Coppersmith multiplier:", str(coppersmith_multiplier))

	if globe:
		coordinates = []
	else:
		if type(parameters_dict["shape_coords"]) is str:
			coordinates = np.array(eval(parameters_dict["shape_coords"]))
		else:
			coordinates = np.array(parameters_dict["shape_coords"])
	pprint.pprint("  Coordinates of region: " + str(list(coordinates)))

	timewindow_length = to_days(timewindow_end - timewindow_start)
	ending = '_mc_' + str(int(round_half_up(mc*10))) + '.csv'

	try:
		fn_prepend = parameters_dict["fn_prepend"]
	except KeyError:
		fn_prepend = ''

	fn_parameters = data_path + fn_prepend + 'parameters.json'
	fn_evol = data_path + fn_prepend + 'parameter_evolution.csv'
	fn_ip = data_path + fn_prepend + 'background_probabilities.csv'
	fn_src = data_path + fn_prepend + 'sources.csv'
	fn_pij = data_path + fn_prepend + 'pij.csv'
	fn_dist = data_path + fn_prepend + 'dist.csv'

	if os.path.exists(fn_parameters):
		print('inversion already done!')
		exit()

	# earth radius in km
	earth_radius = 6.3781e3

	if globe:
		area = earth_radius ** 2 * 4 * np.pi
	else:
		poly = Polygon(coordinates)
		area = polygon_surface(poly)
	print(space, "Region has " + str(area) + " square km")

	# ranges for parameters
	log10_mu_range = (-10, 0)
	log10_k0_range = (-4, 0)
	a_range = (0.01, 5.)
	log10_c_range = (-8, 0)
	omega_range = (-0.99, 1)
	log10_tau_range = (0.01, 5)
	log10_d_range = (-4, 3)
	gamma_range = (0.01, 5.)
	rho_range = (0.01, 5.)

	etas_ranges = log10_mu_range, log10_k0_range, a_range, log10_c_range, omega_range, log10_tau_range, log10_d_range, gamma_range, rho_range

	# start inversion
	print('\n\n')
	print("Reading data..\n")
	df = pd.read_csv(fn_catalog, index_col=0, parse_dates=["time"], dtype={"url": str, "alert": str})
	len_full = len(df)
	if not globe:
		df = gpd.GeoDataFrame(
			df, geometry=gpd.points_from_xy(df.latitude, df.longitude))

		# filter for events in region of interest
		df = df[df.intersects(poly)].copy()
		df.drop("geometry", axis=1, inplace=True)

	print(space, str(len(df)) + " out of " + str(len_full) + " events lie within target region.")

	# filter for events above reference magnitude - delta_m/2
	if delta_m > 0:
		df["magnitude"] = round_half_up(df["magnitude"] / delta_m) * delta_m
	df.query("magnitude>=@m_ref-@delta_m/2", inplace=True)

	print(space, str(len(df)) + " events are above reference magnitude")

	df.query("time >= @ auxiliary_start and time < @ timewindow_end", inplace=True)

	print(space, str(len(df)) + " events are within time window")

	# START

	# assuming discrete (yes/no) detection at all times for initial guessing
	df["current_rate"] = 0
	if distances is None:
		print('\n\n')
		print('Calculating distances...\n')
		print(space, 'assuming perfect detection\n')
		distances = prepare_catalog(
			df,
			coppersmith_multiplier=coppersmith_multiplier,
			timewindow_start=timewindow_start,
			timewindow_end=timewindow_end,
			earth_radius=earth_radius,
			delta_m=delta_m
		)
		# distances.to_csv(fn_dist)
	beta = estimate_beta_tinti(df["magnitude"], mc=m_ref, delta_m=delta_m)
	t_R = 1e-3
	try:
		theta_0 = parameters_dict["theta_0"]
	except KeyError:
		theta_0 = set_initial_values(etas_ranges)

	i = 0
	diff = 1

	evolution = pd.DataFrame()

	while diff > 0.001:
		print('\n\n')

		print("iteration", i)
		print('\n\n')
		print(space, 'inverting ETAS parameters...\n')

		# ETAS parameter inversion
		if i == 0 and mc != m_ref:
			print(space, 'discrete detection with mc =', mc)
			print(space, 'if you gave input distances with weird rates things might get messed up!\n')
			df_0_index = df.query("magnitude >= @mc-@delta_m/2").index.copy()
			beta = estimate_beta_tinti(df.loc[df_0_index, "magnitude"].copy(), mc=mc, delta_m=delta_m)

			new_parameters, _, _, _ = invert_etas_params(
				df.loc[df_0_index, :].copy(),
				beta,
				t_R,
				mc,
				delta_m,
				timewindow_length, timewindow_start, area,
				theta_0=theta_0,
				distances=distances.query("source_id in @df_0_index and target_id in @df_0_index").copy(),
				ranges=etas_ranges,
				verbose=False,
				space=space
			)
			print(2 * space, "parameters with reference magnitude = " + str(mc) + ":\n")
			print(indent(pprint.pformat(
				dict(zip(
					['log10_mu', 'log10_k0', 'a', 'log10_c', 'omega', 'log10_tau', 'log10_d', 'gamma', 'rho'],
					new_parameters
				))
			), 2 * space), '\n')
			print(2 * space, "transforming to reference magnitude = " + str(m_ref) + "...\n")
			new_parameters = parameter_dict2array(
				transform_parameters(
					parameter_array2dict(new_parameters),
					beta=beta,
					delta_m=m_ref-mc
				)
			)
		else:
			new_parameters, sources, df, Pij = invert_etas_params(
				df,
				beta,
				t_R,
				m_ref,
				delta_m,
				timewindow_length, timewindow_start, area,
				theta_0=theta_0,
				distances=distances,
				ranges=etas_ranges,
				verbose=False,
				space=space
			)
		diff = np.abs(new_parameters - theta_0).sum()
		print(2*space, "new ETAS parameters:\n")
		print(indent(pprint.pformat(
			parameter_array2dict(new_parameters)
		), 2*space), '\n')

		print(2*space, "difference to previous:", diff)
		theta_0 = new_parameters

		# detection parameter inversion
		print('\n\n')
		print(space, 'inverting detection parameters...\n')
		det_start = dt.datetime.now()
		p_0 = [beta, t_R]
		p_new, df, distances = rates_and_detection(df, distances, new_parameters, m_ref, delta_m, timewindow_start, p_0=p_0, verbose=False)
		beta, t_R = p_new
		print(2*space, "new detection parameters:", p_new)
		print(2 * space, "detection inversion took:", dt.datetime.now()-det_start)
		i += 1
		evolution = evolution.append(
			pd.Series(list(theta_0) + list(p_0), name=i)
		)
		evolution.to_csv(fn_evol)

	print("CONVERGED AFTER", i, "ITERATIONS")
	print("full inversion took", dt.datetime.now()-start_time)
	evolution.columns = [
		'log10_mu', 'log10_k0', 'a', 'log10_c', 'omega', 'log10_tau', 'log10_d', 'gamma', 'rho', 'beta', 't_R'
	]
	evolution.to_csv(fn_evol)
	df.to_csv(fn_ip)
	sources.to_csv(fn_src)
	if store_pij:
		Pij.to_csv(fn_pij)

	all_info = {
		"has_omori_tau": True,
		"auxiliary_start": str(auxiliary_start),
		"timewindow_start": str(timewindow_start),
		"timewindow_end": str(timewindow_end),
		"timewindow_length": timewindow_length,
		"m_ref": m_ref,
		"mc": mc,
		"delta_m": delta_m,
		"beta": beta,
		"t_R": t_R,
		"n_target_events": len(df),
		"shape_coords": str(list(coordinates)),
		"earth_radius": earth_radius,
		"area": area,
		"coppersmith_multiplier": coppersmith_multiplier,
		"log10_mu_range": log10_mu_range,
		"log10_k0_range": log10_k0_range,
		"a_range": a_range,
		"log10_c_range": log10_c_range,
		"omega_range": omega_range,
		"log10_tau_range": log10_tau_range,
		"log10_d_range": log10_d_range,
		"gamma_range": gamma_range,
		"rho_range": rho_range,
		"ranges": etas_ranges,
		"fn": fn_catalog,
		"fn_ip": fn_ip,
		"fn_src": fn_src,
		"fn_evol": fn_evol,
		"calculation_date": str(dt.datetime.now()),
		"discrete_detection_parameters": str(parameter_array2dict(evolution.iloc[0, :-2])),
		"discrete_detection_beta": evolution.iloc[0, -2],
		"final_parameters": str(parameter_array2dict(new_parameters))
	}

	info_json = json.dumps(all_info)
	f = open(fn_parameters, "w")
	f.write(info_json)
	f.close()
	return evolution
