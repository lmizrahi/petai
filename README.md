# PETAI: Probabilistic, Epidemic-Type Aftershock Incompleteness


### This code was written by Leila Mizrahi with help from Shyam Nandan for the following article:


Leila Mizrahi, Shyam Nandan, Stefan Wiemer 2021;<br/> Embracing Data Incompleteness for Better Earthquake Forecasting.<br/>
_Journal of Geophysical Research: Solid Earth_; doi: https://doi.org/10.1029/2021JB022379<br/>
<br/>
<br/>

To cite the code, please cite the article.<br/>
For more documentation on the code, see the (supporting information of the) article.<br/>
For general ETAS, or ETAS with long-term variations of completeness, see [ETAS](https://github.com/lmizrahi/etas).<br/>
In case of questions or comments, contact me: leila.mizrahi@sed.ethz.ch.
<br/>
<br/>
### Contents:
* inversion.py: PETAI algorithm to estimate ETAS and detection parameters.
* simulation.py: Catalog simulation given ETAS parameters and estimated incompleteness.
* data/synthetic_catalog.csv: example synthetic catalog.
* invert_petai.py: run example PETAI inversion on data/synthetic_catalog.csv.
* simulate_catalog_continuation.py: simulate one example catalog continuation of data/synthetic_catalog.csv, using the parameters and incompleteness estimated when running invert_petai.py.


Just in case, here is my pip freeze:<br/>

* geopandas==0.9.0
* numpy==1.19.1
* pandas==1.1.1
* pynverse==0.1.4.4
* pyproj==3.0.1
* scipy==1.5.2
