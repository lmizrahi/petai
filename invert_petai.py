#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################################################################
# inversion of ETAS and detection parameters (for PETAI)
#
# as described by Mizrahi et al., 2021
# Leila Mizrahi, Shyam Nandan, Stefan Wiemer;
# Embracing Data Incompleteness for Better Earthquake Forecasting.
# Journal of Geophysical Research: Solid Earth.
# doi: https://doi.org/10.1029/2021JB022379
##############################################################################


import datetime as dt

from inversion import invert_etas_and_detecetion_params, parameter_dict2array

if __name__ == '__main__':
	theta_0 = parameter_dict2array({
		'log10_mu': -5.8,
		'log10_k0': -2.6,
		'a': 1.8,
		'log10_c': -2.5,
		'omega': -0.02,
		'log10_tau': 3.5,
		'log10_d': -0.85,
		'gamma': 1.3,
		'rho': 0.66
	})

	inversion_meta = {
		"fn_catalog": "data/synthetic_catalog.csv",
		"data_path": "data/",
		"auxiliary_start": dt.datetime(1970, 1, 1),
		"timewindow_start": dt.datetime(1985, 1, 1),
		"timewindow_end": dt.datetime(2020, 1, 1),
		"theta_0": theta_0,
		"m_ref": 3.0,
		"mc": 3.5,
		"delta_m": 0.1,
		"coppersmith_multiplier": 50,
		"shape_coords": str([
			[20, 100],
			[60, 100],
			[60, 140],
			[20, 140]
		]),
	}

	solution_evolution = invert_etas_and_detecetion_params(
		inversion_meta
	)
