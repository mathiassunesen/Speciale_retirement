Overleaf dokument: https://www.overleaf.com/1322953718gnfxfvttpvbd

This repository provides code for Specialet: "If you go I go - A structural estimation of joint retirement in Danish households" by Frederik Kristensen og Mathias Sunesen.

Dependencies:
https://github.com/NumEconCopenhagen/ConsumptionSaving: pip install git+https://github.com/NumEconCopenhagen/ConsumptionSaving
https://pypi.org/project/numba/: $ conda install numba

The code is found in "main" and is structured in the following way:

Folder:
estimates:			estimated parameters
figs: 				figures
SASdata: 			data (moments, wealth, etc.)
 
Notebooks:
Calibration:			calibration of survival probabilities
Couples:			analysis of couple model
Experiments:			counterfactual policy simulations
MSM_real_data:			estimation on real data
MSM_sim_data:			estimation on simulated data from the model
Singles:			analysis of single model

Python files:
egm:				egm step (part of solving the model)
figs:				functions for plotting
funs:				misc functions (Gauss Hermite, logsum etc.)
last_period:			solving the last period of the model
Model:				class for the model
post_decision:			post decision step (part of solving the model)
setup:				set up the model (tax system, retirement system, precomputations etc.)
simulate:			functions for simulating the model
SimulatedMinimumDistance:	functions for estimation (moments functions, optimizer etc.)
solution:			wrapper function for solving the model
transitions:			functions for tax system and precomputations etc.
utility:			utility functions


