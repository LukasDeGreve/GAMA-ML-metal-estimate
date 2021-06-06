"""
Code to create the results from chapter 3.

There are few comments (sorry), but the code should be quite straightforward
The majority of the code is matplotlib magic
For info on all the dataprocessing stuff, check those functions as they do have extended documentation
Finally there are some fits with scipy. Check the scipy documentation if there are any unclear things
"""

#TODO insets toevoegen met aantal galaxies per plot (voor histo's)
#TODO mediaan berekenen en gebruiken in besprekingen

from tokenize import Exponent
from numpy.core.fromnumeric import shape
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
from scipy.sparse import data
import Plots
import Dataprocessing as dp
import Machine_learning as ml
from scipy.optimize import curve_fit
from scipy import odr
import sklearn
import scipy as sc


bands_list = ["FUV_flux", "NUV_flux", "u_flux", "g_flux", "r_flux", "i_flux", "z_flux", "X_flux", "Y_flux", "J_flux", "H_flux", "K_flux"
,"W1_flux", "W2_flux", "W3_flux", "W4_flux", "P100_flux", "P160_flux", "S250_flux", "S350_flux", "S500_flux"]
SDSS_list = ["u_flux", "g_flux", "r_flux", "i_flux", "z_flux"]
WISE_list = ["W1_flux", "W2_flux", "W3_flux", "W4_flux"]
Herschel_list = ["P100_flux", "P160_flux", "S250_flux", "S350_flux", "S500_flux"]
GALEX_list = ["FUV_flux", "NUV_flux"]
VIKING_list = ["X_flux", "Y_flux", "J_flux", "H_flux", "K_flux"]


simple = pd.read_csv("/run/media/lukasdg/Seagate/Documents/School/2e master/Thesis/Data/GaussFitSimple.csv")
complex = pd.read_csv("/run/media/lukasdg/Seagate/Documents/School/2e master/Thesis/Data/GaussFitComplex.csv")
bands = pd.read_csv("/run/media/lukasdg/Seagate/Documents/School/2e master/Thesis/Data/LambdarCat.csv")
mass = pd.read_csv("/run/media/lukasdg/Seagate/Documents/School/2e master/Thesis/Data/StellarMassesLambdar.csv")
sfr = pd.read_csv("/run/media/lukasdg/Seagate/Documents/School/2e master/Thesis/Data/MagPhys.csv")

colours = Plots.set_defaults()


# mass
plt.figure("mass")
print(mass.shape)
mass = dp.AddMass(mass, verbose=True)
print(mass.shape)
mass_row = mass['MASS'].squeeze()
masslogbins = np.logspace(8, 12, 20)
plt.subplot(311)
plt.hist(mass_row, bins=20, rwidth=0.95, range=(7.5, 12.5))
box = dict(facecolor='none', edgecolor='lightgray', boxstyle='round')
plt.annotate("total galaxies: {}".format(len(mass_row)), xy=(0.025, 0.825), xycoords='axes fraction', bbox = box)
print("median mass: {}".format(np.median(mass_row)))


# metallicity
plt.figure("mets")
lines = simple.merge(complex, on=["CATAID", "SPECID"])
print(lines.shape)
lines2 = lines
# always use KeepBestGalaxy first, before removing stuff with other functions
lines = dp.KeepBestGalaxy(lines, verbose=True)
print(lines.shape)
lines = dp.SelectSurvey(lines, ("GAMA", "SDSS"), verbose=True)
print(lines.shape)
lines = dp.RemoveBadFit(lines, method="PG16", SNR=3, verbose=True)
print(lines.shape)
lines = dp.AddMetallicity(lines, method="PG16", SNR=3, verbose=True)
print(lines.shape)
metal_row = lines["METAL"].squeeze()
plt.subplot(311)
plt.hist(metal_row, bins=20, rwidth=0.95, range=(7.5, 9.5))
plt.annotate("total galaxies: {}".format(len(metal_row)), xy=(0.025, 0.825), xycoords='axes fraction', bbox = box)
print("median metallicity: {}".format(np.median(metal_row)))

count = 0
for i in metal_row:
    if i > 8.7:
        count += 1
print (count, len(metal_row))
exit()

# broadband fluxes
plt.figure("flux")
print(bands.shape)
plt.subplot(211)
Plots.CheckMissingData(bands, bands_list, show=False, labels=False, verbose=True)
plt.xticks([])
plt.annotate("total galaxies: {}".format(bands.shape[0]), xy=(0.025, 0.825), xycoords='axes fraction', bbox = box)
plt.ylabel("Number of galaxies")

# sfr
plt.figure("sfr")
print(sfr.shape)
sfr_row = sfr["SFR_0_1Gyr_best_fit"].squeeze()
sfrlogbins = np.logspace(-4.5, 2.5, 20)
plt.subplot(311)
plt.hist(sfr_row, bins=sfrlogbins, rwidth=0.95)
plt.xscale("log")
plt.annotate("total galaxies: {}".format(len(sfr_row)), xy=(0.025, 0.825), xycoords='axes fraction', bbox = box)
print("median SFR: {}".format(np.median(sfr_row)))


# gotta (catch) merge 'em all
master = bands.merge(lines[["CATAID", "METAL", "Z", "OIIIR_FLUX_x", "HB_FLUX_y", "SIIB_FLUX", "HA_FLUX_y"]], on="CATAID")
print(master.shape)
master = master.merge(mass[["CATAID", "MASS"]], on="CATAID")
print(master.shape)
master = master.merge(sfr[["CATAID", "SFR_0_1Gyr_best_fit"]], on="CATAID")
print(master.shape)

# pegged parameters stuff
lines2 = dp.FindBadGalaxies(lines2, verbose=True)
bad = lines2.merge(mass[["CATAID", "MASS"]], on="CATAID")
bad = bad.merge(sfr[["CATAID", "SFR_0_1Gyr_best_fit"]], on="CATAID")
bad = bad.sample(n=master.shape[0], random_state=1)
plt.figure("test")
x, y, c = Plots.estimate_density(bad["MASS"].squeeze(), np.log10(bad["SFR_0_1Gyr_best_fit"].squeeze()))
plt.scatter(x, y, c=c, alpha=0.5)
plt.xlabel(r"log(Mass) (log($M_\odot$))")
plt.ylabel(r"log(SFR) (log($M_\odot$/yr))")
plt.xlim(6.5, 13)
plt.ylim(-4.5, 3)
plt.title("M-SFR relation of galaxies with pegged parameters")


# AGN removal
master = dp.RemoveAGN(master, plot=True, verbose=True)
print(master.shape)


# Plots using merged data
plt.figure("mass")
mass_row = master['MASS'].squeeze()
plt.subplot(312)
plt.ylabel("Number of galaxies")
plt.hist(mass_row, bins=20, rwidth=0.95, range=(7.5, 12.5))
plt.annotate("total galaxies: {}".format(len(mass_row)), xy=(0.025, 0.825), xycoords='axes fraction', bbox = box)
print("median mass: {}".format(np.median(mass_row)))

plt.figure("mets")
mets_row = master['METAL'].squeeze()
plt.subplot(312)
plt.ylabel("Number of galaxies")
plt.hist(mets_row, bins=20, rwidth=0.95, range=(7.5, 9.5))
plt.annotate("total galaxies: {}".format(len(mets_row)), xy=(0.025, 0.825), xycoords='axes fraction', bbox = box)
print("median metallicity: {}".format(np.median(mets_row)))

plt.figure("flux")
plt.subplot(212)
Plots.CheckMissingData(master, bands_list, show=False, labels=True, verbose=True)
plt.annotate("total galaxies: {}".format(master.shape[0]), xy=(0.025, 0.825), xycoords='axes fraction', bbox = box)
plt.title("")

plt.figure("sfr")
sfr_row = master["SFR_0_1Gyr_best_fit"].squeeze()
plt.subplot(312)
plt.hist(sfr_row, bins=sfrlogbins, rwidth=0.95)
plt.xscale("log")
plt.ylabel("Number of galaxies")
plt.annotate("total galaxies: {}".format(len(sfr_row)), xy=(0.025, 0.825), xycoords='axes fraction', bbox = box)
print("median SFR: {}".format(np.median(sfr_row)))


# MZR
def mzrfit(M, Z_0, M_0, gamma, beta):
    exponent = -beta * (M - M_0)
    return Z_0 - gamma/beta * np.log10(1 + np.power(10, exponent))

popt, pcov = curve_fit(mzrfit, mass_row, mets_row, bounds=([5., 5., 0., 0.], [15., 15., 2., 5.]))
print(popt)
xline = np.linspace(6, 13, 50)
yline = mzrfit(xline, *popt)
yline2 = mzrfit(xline, 8.793, 10.02, 0.28, 1.2)
rmse = sklearn.metrics.mean_squared_error(mzrfit(mass_row, *popt), mets_row, squared=False)
r2 = sklearn.metrics.r2_score(mzrfit(mass_row, *popt), mets_row)
spear = sc.stats.spearmanr(mzrfit(mass_row, *popt), mets_row)
print("MZR rmse: {:.2f}, r2: {:.2f}".format(rmse, r2))

plt.figure("mzr")
plt.title("Mass metallicity relation")
plt.subplot(121)
x, y, c = Plots.estimate_density(mass_row, mets_row)
plt.scatter(x, y, c=c, alpha=0.5)
plt.plot(xline, yline, color='red', label="This work")
plt.plot(xline, yline2, color='green', label='Curti et al. 2019')
plt.plot(xline, -2.2 + 1.95*xline-0.085*xline*xline, color='lightblue', label='Foster et al. 2012')
plt.xlabel(r"log(Mass) (log($M_\odot$))")
plt.ylabel("12 + log(O/H)")
plt.xlim(7, 12.5)
plt.ylim(7.45, 9.5)
box = dict(facecolor='none', edgecolor='lightgray', boxstyle='round')
plt.text(7.2, 9.2, 'rmse: {:.2f} \n' r'$R^2$' ': {:.2f} \n' r'$\rho$' ': {:.2f}'.format(rmse, r2, spear[0]), bbox=box)
plt.legend(loc='lower right')


# other relations
plt.figure("msfrr")
plt.title("Mass SFR relation")
ax = plt.subplot(121)
def firstorder(p, x):
    return p[0]*x + p[1]
linear_model = odr.Model(firstorder)
data = odr.Data(mass_row, np.log10(sfr_row))
myodr = odr.ODR(data, linear_model, beta0=[1, -10])
out = myodr.run()
print(out.beta)
yline = firstorder(out.beta, xline)
rmse = sklearn.metrics.mean_squared_error(firstorder(out.beta, mass_row), np.log10(sfr_row), squared=False)
r2 = sklearn.metrics.r2_score(firstorder(out.beta, mass_row), np.log10(sfr_row))
spear = sc.stats.spearmanr(firstorder(out.beta, mass_row), np.log10(sfr_row))
print("mSFRr rmse: {:.2f}, r2: {:.2f}".format(rmse, r2))

x, y, c = Plots.estimate_density(bad["MASS"].squeeze(), np.log10(bad["SFR_0_1Gyr_best_fit"].squeeze()))
ax.scatter(x, y, c=c, alpha=0.25, cmap=plt.cm.viridis)
x, y, c = Plots.estimate_density(mass_row, np.log10(sfr_row))
ax.scatter(x, y, c=c, alpha=0.4, cmap=plt.cm.inferno)
plt.plot(xline, yline, color="red", label="This work")
plt.plot(xline, firstorder([1.304, -12.98], xline), color="green", label="Foster et al. 2012")
plt.xlabel(r"log(Mass) (log($M_\odot$))")
plt.ylabel(r"log(SFR) (log($M_\odot$/yr))")
plt.xlim(6.5, 13)
plt.ylim(-4.5, 3)
plt.text(6.7, 2.1, 'rmse: {:.2f} \n' r'$R^2$' ': {:.2f} \n' r'$\rho$' ': {:.2f}'.format(rmse, r2, spear[0]), bbox=box)
plt.legend(loc='lower right')


plt.figure("sfrzr")
plt.title("SFR metallicity relation of the full dataset")
plt.subplot(121)
data = odr.Data(np.log10(sfr_row), mets_row)
myodr = odr.ODR(data, linear_model, beta0=[0.25, 8.5])
out = myodr.run()
print(out.beta)
xline2 = np.linspace(-5, 3)
yline = firstorder(out.beta, xline2)
rmse = sklearn.metrics.mean_squared_error(firstorder(out.beta, np.log10(sfr_row)), mets_row, squared=False)
r2 = sklearn.metrics.r2_score(firstorder(out.beta, np.log10(sfr_row)), mets_row)
spear = sc.stats.spearmanr(firstorder(out.beta, np.log10(sfr_row)), mets_row)
print("SFRzr rmse: {:.2f}, r2: {:.2f}".format(rmse, r2))

x, y, c = Plots.estimate_density(np.log10(sfr_row), mets_row)   
plt.scatter(x, y, c=c, alpha=0.5)
plt.plot(xline2, yline)
plt.xlabel(r"log(SFR) (log($M_\odot$/yr))")
plt.ylabel("12 + log(O/H)")
plt.xlim(-4.5, 3)
plt.ylim(7.45, 9.5)
plt.text(-4.2, 9.2, 'rmse: {:.2f} \n' r'$R^2$' ': {:.2f} \n' r'$\rho$' ': {:.2f}'.format(rmse, r2, spear[0]), bbox=box)


# FMR
def fmrfit(MSFR, Z_0, m_0, m_1, gamma, beta):
    exponent = -beta*(MSFR[0] -m_0 - m_1*MSFR[1])   
    return Z_0 - gamma/beta * np.log10(1+np.power(10, exponent))

popt, pcov = curve_fit(fmrfit, (mass_row, np.log10(sfr_row)), mets_row, p0=(8.779, 10.11, 0.56, 0.31, 2.1))
print(popt)
yline = fmrfit((xline, xline2), *popt)
rmse = sklearn.metrics.mean_squared_error(fmrfit((mass_row, np.log10(sfr_row)), *popt), mets_row, squared=False)
r2 = sklearn.metrics.r2_score(fmrfit((mass_row, np.log10(sfr_row)), *popt), mets_row)
print("FMR rsme: {:.2f}, r2: {:.2f}".format(rmse, r2))

fig = plt.figure("fmr")
ax = fig.add_subplot(projection='3d')
ax.scatter(np.log10(sfr_row), mass_row, mets_row, alpha=1, marker='.')
X, Y = np.meshgrid(xline, xline2)
zs = np.array(fmrfit((np.ravel(X), np.ravel(Y)), *popt))
Z = zs.reshape(X.shape)
ax.plot_surface(X, Y, Z, antialiased=False, linewidth=0.5)
ax.set_xlabel(r"log(SFR) (log($M_\odot$/yr))")
ax.set_xlim(-2.5, 2.5)
ax.set_ylabel(r"log(Mass) (log($M_\odot$))")
ax.set_ylim(7, 12)
ax.set_zlabel("12 + log(O/H)")
ax.set_zlim(7.5, 9.5)
plt.title("Fundamental metallicity relation of the full dataset")

plt.figure("2dfmr")
plt.title("2D fundamental metallicity relation")
axes = plt.subplot(121)
subpos = [0.12, 0.7, 0.28, 0.28]
subpos2 = [0.35, 0.1, 0.3, 0.3]
subax = Plots.add_subplot_axes(axes, subpos)
subax2 = Plots.add_subplot_axes(axes, subpos2)
M_list = []
gamma_list = []
axes.set_prop_cycle(cycler('color', plt.cm.inferno(np.linspace(0.07, 1, 11))))


sfr_range = np.linspace(-.8, 1.4, 11, False)
for i in sfr_range:
    FMR_list = np.array([mass_row, mets_row, np.log10(sfr_row)])
    mask = np.logical_and(FMR_list[2] > i, FMR_list[2] <= i+0.2)
    FMR_list = np.swapaxes(FMR_list, 0, 1)
    FMR_list = FMR_list[mask]
    popt, pcov = curve_fit(mzrfit, FMR_list[:, 0], FMR_list[:, 1], bounds=([5., 5., 0., 0.], [15., 15., 2., 5.]))
    print(popt)
    M_list.append(popt[1])
    gamma_list.append(popt[2])
    rmse = sklearn.metrics.mean_squared_error(mzrfit(mass_row, *popt), mets_row, squared=False)
    print("MZR rmse: {} for sfr : {:.2f}".format(rmse, i + 0.1))
    yline = mzrfit(xline, *popt)
    labelstr = "{:.1f}" r"$M_\odot$/yr".format(i+0.1)
    axes.plot(xline, yline, label = labelstr)
subax.plot(sfr_range, M_list)
subax.set_xlabel(r"log(SFR) (log($M_\odot$/yr))", fontsize='x-small')
subax.set_ylabel(r"$M_0$ (log($M_\odot$))", fontsize='x-small')
subax2.plot(sfr_range, gamma_list)
subax2.set_xlabel(r"log(SFR) (log($M_\odot$/yr))", fontsize='x-small')
subax2.set_ylabel(r"$\gamma$", fontsize='x-small')
axes.set_xlim(7, 12.5)
axes.set_ylim(7.3, 9.1)
axes.set_xlabel(r"log(Mass) (log($M_\odot$))")
axes.set_ylabel("12 + log(O/H)")
axes.legend(loc='lower right')


Plots.set_defaults()
# remove missing bands and repeat
master = dp.RemoveMissing(master, bands_list)
print(master.shape)

plt.figure("mass")
mass_row = master['MASS'].squeeze()
plt.subplot(313)
plt.hist(mass_row, bins=20, rwidth=0.95, range=(7.5, 12.5))
plt.xlabel(r"log(Mass) (log($M_\odot$))")
plt.annotate("total galaxies: {}".format(len(mass_row)), xy=(0.025, 0.825), xycoords='axes fraction', bbox = box)
print("median mass: {}".format(np.median(mass_row)))

plt.figure("mets")
mets_row = master['METAL'].squeeze()
plt.subplot(313)
plt.hist(mets_row, bins=20, rwidth=0.95, range=(7.5, 9.5))
plt.xlabel("12 + log(O/H)")
plt.annotate("total galaxies: {}".format(len(mets_row)), xy=(0.025, 0.825), xycoords='axes fraction', bbox = box)
print("median metallicity: {}".format(np.median(mets_row)))

plt.figure("sfr")
sfr_row = master["SFR_0_1Gyr_best_fit"].squeeze()
plt.subplot(313)
plt.hist(sfr_row, bins=sfrlogbins, rwidth=0.95)
plt.xscale("log")
plt.xlabel(r"SFR ($M_\odot$/yr)")
plt.annotate("total galaxies: {}".format(len(sfr_row)), xy=(0.025, 0.825), xycoords='axes fraction', bbox = box)
print("median SFR: {}".format(np.median(sfr_row)))


# MZR
popt, pcov = curve_fit(mzrfit, mass_row, mets_row, bounds=([5., 5., 0., 0.], [15., 15., 2., 5.]))
print(popt)
yline = mzrfit(xline, *popt)
yline2 = mzrfit(xline, 8.793, 10.02, 0.28, 1.2)
rmse = sklearn.metrics.mean_squared_error(mzrfit(mass_row, *popt), mets_row, squared=False)
r2 = sklearn.metrics.r2_score(mzrfit(mass_row, *popt), mets_row)
spear = sc.stats.spearmanr(mzrfit(mass_row, *popt), mets_row)
print("MZR rmse: {:.2f}, r2: {:.2f}".format(rmse, r2))

plt.figure("mzr")
plt.subplot(122)
x, y, c = Plots.estimate_density(mass_row, mets_row)
plt.scatter(x, y, c=c, alpha=0.5)
plt.plot(xline, yline, color='red', label="This work")
plt.plot(xline, yline2, color='green', label='Curti et al. 2019')
plt.plot(xline, -2.2 + 1.95*xline-0.085*xline*xline, color='lightblue', label='Foster et al. 2012')
plt.xlabel(r"log(Mass) (log($M_\odot$))")
plt.xlim(7, 12.5)
plt.ylim(7.45, 9.5)
plt.text(7.2, 9.2, 'rmse: {:.2f} \n' r'$R^2$' ': {:.2f} \n' r'$\rho$' ': {:.2f}'.format(rmse, r2, spear[0]), bbox=box)
plt.legend(loc='lower right')


# other relations
plt.figure("msfrr")
plt.subplot(122)
data = odr.Data(mass_row, np.log10(sfr_row))
myodr = odr.ODR(data, linear_model, beta0=[1, -10])
out = myodr.run()
print(out.beta)
yline = firstorder(out.beta, xline)
rmse = sklearn.metrics.mean_squared_error(firstorder(out.beta, mass_row), np.log10(sfr_row), squared=False)
r2 = sklearn.metrics.r2_score(firstorder(out.beta, mass_row), np.log10(sfr_row))
spear = sc.stats.spearmanr(firstorder(out.beta, mass_row), np.log10(sfr_row))
print("mSFRr rmse: {:.2f}, r2: {:.2f}".format(rmse, r2))

x, y, c = Plots.estimate_density(mass_row, np.log10(sfr_row))
plt.scatter(x, y, c=c, alpha=0.5)
plt.plot(xline, yline, color="red", label="This work")
plt.plot(xline, firstorder([1.304, -12.98], xline), color="green", label="Foster et al. 2012")
plt.xlabel(r"log(Mass) (log($M_\odot$))")
plt.xlim(6.5, 13)
plt.ylim(-4.5, 3)
plt.text(6.7, 2.1, 'rmse: {:.2f} \n' r'$R^2$' ': {:.2f} \n' r'$\rho$' ': {:.2f}'.format(rmse, r2, spear[0]), bbox=box)
plt.legend(loc='lower right')


plt.figure("sfrzr")
plt.subplot(122)
data = odr.Data(np.log10(sfr_row), mets_row)
myodr = odr.ODR(data, linear_model, beta0=[0.25, 8.5])
out = myodr.run()
print(out.beta)
xline2 = np.linspace(-5, 3)
yline = firstorder(out.beta, xline2)
rmse = sklearn.metrics.mean_squared_error(firstorder(out.beta, np.log10(sfr_row)), mets_row, squared=False)
r2 = sklearn.metrics.r2_score(firstorder(out.beta, np.log10(sfr_row)), mets_row)
spear = sc.stats.spearmanr(firstorder(out.beta, np.log10(sfr_row)), mets_row)
print("SFRzr rmse: {:.2f}, r2: {:.2f}".format(rmse, r2))

x, y, c = Plots.estimate_density(np.log10(sfr_row), mets_row)   
plt.scatter(x, y, c=c, alpha=0.5)
plt.plot(xline2, yline)
plt.xlabel(r"log(SFR) (log($M_\odot$/yr))")
plt.xlim(-4.5, 3)
plt.ylim(7.45, 9.5)
plt.text(-4.2, 9.2, 'rmse: {:.2f} \n' r'$R^2$' ': {:.2f} \n' r'$\rho$' ': {:.2f}'.format(rmse, r2, spear[0]), bbox=box)


# FMR
popt, pcov = curve_fit(fmrfit, (mass_row, np.log10(sfr_row)), mets_row, p0=(8.779, 10.11, 0.56, 0.31, 2.1))
print(popt)
yline = fmrfit((xline, xline2), *popt)
rmse = sklearn.metrics.mean_squared_error(fmrfit((mass_row, np.log10(sfr_row)), *popt), mets_row, squared=False)
r2 = sklearn.metrics.r2_score(fmrfit((mass_row, np.log10(sfr_row)), *popt), mets_row)
print("FMR rsme: {:.2f}, r2: {:.2f}".format(rmse, r2))
plt.figure("2dfmr")
axes = plt.subplot(122)
subax = Plots.add_subplot_axes(axes, subpos)
subax2 = Plots.add_subplot_axes(axes, subpos2)
M_list = []
gamma_list = []
axes.set_prop_cycle(cycler('color', plt.cm.inferno(np.linspace(0.07, 1, 11))))

sfr_range = np.linspace(-.8, 1.4, 11, False)
for i in sfr_range:
    FMR_list = np.array([mass_row, mets_row, np.log10(sfr_row)])
    mask = np.logical_and(FMR_list[2] > i, FMR_list[2] <= i+0.2)
    FMR_list = np.swapaxes(FMR_list, 0, 1)
    FMR_list = FMR_list[mask]
    popt, pcov = curve_fit(mzrfit, FMR_list[:, 0], FMR_list[:, 1], bounds=([5., 5., 0., 0.], [15., 15., 2., 5.]))
    print(popt)
    M_list.append(popt[1])
    gamma_list.append(popt[2])
    rmse = sklearn.metrics.mean_squared_error(mzrfit(mass_row, *popt), mets_row, squared=False)
    print("MZR rmse: {} for sfr : {:.2f}".format(rmse, i + 0.1))
    yline = mzrfit(xline, *popt)
    labelstr = "{:.1f}" r"$M_\odot$/yr".format(i+0.1)
    axes.plot(xline, yline, label = labelstr)
subax.plot(sfr_range, M_list)
subax.set_xlabel(r"log(SFR) (log($M_\odot$/yr))", fontsize='x-small')
subax.set_ylabel(r"$M_0$ (log($M_\odot$))", fontsize='x-small')
subax2.plot(sfr_range, gamma_list)
subax2.set_xlabel(r"log(SFR) (log($M_\odot$/yr))", fontsize='x-small')
subax2.set_ylabel(r"$\gamma$", fontsize='x-small')
axes.set_xlim(7, 12.5)
axes.set_ylim(7.3, 9.1)
axes.set_xlabel(r"log(Mass) (log($M_\odot$))")
axes.legend(loc='lower right')


plt.show()