# GAMA-ML-metal-estimate

Github repository accompanying the master thesis ''Estimating metallicity using machine learning''. A link will be added once the paper is added to the UGent library.

Machine learning techniques were applied on the GAMA dataset to estimate the metallicity of some 4000 galaxies. Most results of the paper can be replicated with this code, but the settings need to be manually changed. Part of the code can be used to obtain gas phase metallicities of around 10000-45000 galaxies, depeding on data selection and SN criteria. The code should be fairly well documented, but if any questions remain, please contact me at lukas.degreve@hotmail.com. The code was written in python 3.8.5 and uses the sklearn v0.23 library for machine learning. Other used libraries are numpy, pandas, matplotlib and scipy.The CIGALE output is not included.

Dataprocessing.py, machine_learning.py and Plots.py contain the function needed for main.py and GAMA relations.py. GAMA relations.py reproduces all plots and some extras seen in chapter 3 of the thesis. Main.py is the metallicity estimator and is used to create the plots in chapter 4.

Additional downloads of the GAMA DR3 data are required. They can be found here: http://www.gama-survey.org/dr3/schema/. The main program requires the GaussFitSimple, GaussFitComplex, StellarMassesLambdar and LambdarCat DMU's, GAMA relations.py additionally requires the MagPhys DMU. These databases need to be either downloaded as or converted to a csv file. Conversion from .fits to csv can be done by for example the TOPCAT program, or a python script using the pandas library.

I have no idea how copyright works, so please cite my thesis and/or refer to this github page if you use or modify this code in further research
