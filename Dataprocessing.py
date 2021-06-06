import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
import Plots


# All functions needed to end up with a usable dataframe for machine learning
# this includes functions that add metallicity, crossmatch the data, and remove a variety of unwanted data
# Some legacy stuff involving crossmatching galaxies with the SDSS DR7 catalogue are still left but not used in the paper


def RemoveMissing(dataframe, bands, verbose=False):
    """
    Removes all entries where one or more fluxes are missing from the given bands

    Parameters
    ----------
    dataframe: Pandas Dataframe
        Dataframe which contains the data that has to be cleaned
    bands: (numpy) array
        array containing the column names that have to be checked
    verbose: bool
        when true, prints what the program is currently doing and additional outputs

    Returns
    -------
    dataframe: Pandas Dataframe
        dataframe where all rows with missing data are removed
    """
    
    print("checking data...") if verbose else None

    del_list = []
    for i, row in dataframe.iterrows():
        if np.count_nonzero(row[bands]) != len(row[bands]):
            del_list.append(i)

    print("Rows removed: {}".format(len(del_list))) if verbose else None

    return(dataframe.drop(del_list))



def SelectSurvey(dataframe, survey, survey_code=False, verbose=False):
    '''
    Removes all data not from the surveys designated by the parameters survey
    This is for example usefull to only keep SDSS and GAMA data, as those are the only
    two surveys with callibrated spectra
    
    Parameters
    ----------
    dataframe: Pandas dataframe
        dataframe containing the data
    survey: (numpy) array
        array containing the names or codes of the surveys you want to keep
    survey_code: bool
        if True, use survey codes instead of survey names
    verbose: bool
        if True, prints status messages and additional outputs

    Returns
    -------
    dataframe: Pandas dataframe
        dataframe with only the rows of the given surveys are left

    '''

    print("Checking data...") if verbose else None

    del_list = []

    if survey_code:
        for i, row in dataframe.iterrows():
            if row["SURVEY_CODE"] not in survey:
                del_list.append(i)
    else:
        for i, row in dataframe.iterrows():
            if row["SURVEY"] not in survey:
                del_list.append(i)

    dataframe = dataframe.drop(del_list)
    print("Removed all data from other surveys!") if verbose else None
    print("Rows removed: {}".format(len(del_list))) if verbose else None
    
    return dataframe


def KeepBestGalaxy(dataframe, verbose=False):
    '''
    Some objects have multiple spectra. This program keeps the best one by looking at
    the SN ratio along the continuum
    !!! USE BEFORE REMOVING STUFF FROM DATAFRAMES WITH OTHER FUNCTIONS !!!
    because of the way the code is written, removing rows results in an out of range error

    Parameters
    ----------
    dataframe: Pandas dataframe
        dataframe containing the data. Needs a column "SN" with some kind of signal to noise 
        ratio and a column "CATAID" containing the name of the object
    verbose: bool
        if True, prints status messages and additional outputs
    
    Returns
    -------
    dataframe: pandas dataframe
        dataframe where only the best spectrum of each galaxy is kept

    '''

    print("Checking data...") if verbose else None

    index_list = [] # list of indices of the data that gets removed
    best_SN = 0
    best_index = 0 # index of the current best SN candidate
    SN_index = dataframe.columns.get_loc("SN") # index of the "SN" column, needed for iloc
    CATA_index = dataframe.columns.get_loc("CATAID") # Same as above, now for CATAID

    for i, row in dataframe.iloc[1:].iterrows():
        if row["CATAID"] == dataframe.iloc[i-1, CATA_index]:
            if row["SN"] > best_SN:
                index_list.append(best_index)
                best_SN = row["SN"]
                best_index = i
            else:
                index_list.append(i)
        else:
            best_SN = row["SN"]
            best_index = i

    dataframe = dataframe.drop(index_list)
    
    print("Removed double data!") if verbose else None
    print("Galaxies with multiple spectra: {}".format(len(index_list))) if verbose else None

    return (dataframe)


def RemoveBadFit(dataframe, method="Tremonti04", SNR=3, verbose=False):
    """
    Checks the NPEG and FITFAIL parameters of the lines relevant to the method used
    and removes data where these parameters indicate a bad fit
    Furthermore this method also checks the line S/N and removes data where this is < SNR
    For more information, check the description of the DMU at http://www.gama-survey.org/dr3/schema/dmu.php?id=8
    or the paper accompanying the DMU: Driver et al. (2018)

    Parameters
    ----------
    dataframe: pandas dataframes
        dataframe with the line fluxes. must contain both the simple and complex fits from the DMU,
        merged with the standard pandas merge function
    Method: Tremonti04, PG16
        Method to estimate metallicity. Check the respective papers for more info
    SNR: float
        ratio of line flux and error under which measurements should be discarded. Using anything under 2~3
        is not advised
    Verbose: bool
        if True, prints status messages and additional outputs
        The diagnostics at the end of the program are not entirely correct:if both 
        the S/N is insufficient and parameters are pegged, the line is only counted once
        and added to the amount of lines with pegged parameters, not those with bad S/N

    Returns
    -------
    dataframe: pandas dataframe
        the same dataframe as inputted, now with al unwanted data removed
    """
    fitfail = 0
    pegged = 0
    SNrat = 0
    other = 0

    if method == "Tremonti04":  # Tremonti et al (2004), ApJ 613, 898

        print("Checking data for using Tremonti et al (2004)...") if verbose else None
        del_list = []

        for i, rows in dataframe.iterrows():
            if rows["OII_FITFAIL"] == 1 or ["HB_FITFAIL_y"] == 1:
                del_list.append(i)
                fitfail += 1
            elif rows["OIIR_NPEG"] > 0 or rows["OIIIR_NPEG_x"] > 0 or rows["HB_NPEG_y"] > 0:
                del_list.append(i)
                pegged += 1
            elif SNR*rows["OIIR_FLUX_ERR"] > rows["OIIR_FLUX"] or SNR*rows["OIIIR_FLUX_ERR_x"] > rows["OIIIR_FLUX_x"] or SNR*rows["OIIIB_FLUX_ERR_x"] >  rows["OIIIB_FLUX_x"] or SNR*rows["HB_FLUX_ERR_y"] > rows["HB_FLUX_y"]:
                del_list.append(i)
                SNrat += 1

        dataframe = dataframe.drop(del_list)


    elif method == "PG16":   # Pilyugin and Grebel (2016), MNRAS 457, 3678

        print("Checking data for using Pilyugin and Grebel (2016)...") if verbose else None
        del_list = []

        # First check the lines both recipes need (R and S recipe)
        # Not everything is in the same if statement for clarity and to prevent very long lines
        for i, rows in dataframe.iterrows():
            if rows["HB_FITFAIL_x"] == 1 or rows["HA_FITFAIL_x"] == 1 or rows["HB_FITFAIL_y"] == 1:
                del_list.append(i)
                fitfail += 1
            elif rows["OIIIR_NPEG_x"] > 0 or rows["OIIIB_NPEG_x"] > 0 or rows["NIIR_NPEG_x"] > 0 or rows["NIIB_NPEG_x"] > 0 or rows["HB_NPEG_y"] > 0:
                del_list.append(i)
                pegged += 1
            elif SNR*rows["OIIIR_FLUX_ERR_x"] > rows["OIIIR_FLUX_x"] or SNR*rows["OIIIB_FLUX_ERR_x"] > rows["OIIIB_FLUX_x"] or SNR*rows["NIIB_FLUX_ERR_x"] > rows["NIIB_FLUX_x"] or SNR*rows["NIIR_FLUX_ERR_x"] > rows["NIIR_FLUX_x"] or SNR*rows["HB_FLUX_ERR_y"] > rows["HB_FLUX_y"]:
                del_list.append(i)
                SNrat += 1

            # Now check if there is good enough data for at least 1 recipe
            # Due to the nature of these recipes, it is quite difficult to make verbose work while also being efficient
            # I have opted for the most efficient method, as a consequence making verbose a bit more nondescriptive
            noR2 = rows["OII_FITFAIL"] == 1 or rows["OIIR_NPEG"] > 0 or rows["OIIB_NPEG"] > 0 or SNR*rows["OIIR_FLUX_ERR"] > rows["OIIR_FLUX"]
            noS2 = rows["SII_FITFAIL"] == 1 or rows["SIIR_NPEG"] > 0 or rows["SIIB_NPEG"] > 0 or SNR*rows["SIIB_FLUX_ERR"] > rows["SIIB_FLUX"] or SNR*rows["SIIR_FLUX_ERR"] > rows["SIIR_FLUX"]

            if noR2 and noS2:
                del_list.append(i)
                other += 1

        dataframe = dataframe.drop(del_list)

    print("Removed bad fits!")
    if verbose:
        print("Number of spectra with failed fits: {}".format(fitfail))
        print("Number of spectra which have pegged at the boundaries : {}".format(pegged))
        print("Number of spectra with lines S/N < {}: {}".format(SNR, SNrat))
        print("Number of bad spectra due to other causes: {}".format(other))

    return(dataframe)


def AddMetallicity(dataframe, method="Tremonti04", SNR=3, verbose=False):
    """
    Pretty much what the title say. Choose the method and row will be added containing metallicity
    The Tremonti method is simpler, but is a fit based on photoionization models and is thus 
    expected to perform worse than the PG16 method (and is in general quite bad), which is based on electron temperature
    PG16 is also able to deal with a bigger range of metallicities and resolves the double-valuedness
    of the R23 line. Abnormal metallicities (far outside the callibration range of the method) are discarded

    Parameters
    ----------
    dataframe: Pandas Dataframe
        single dataframe consisting of the simple and complex line fits, preferably merged with 
        the pandas.merge function
    method: Tremonti04, PG16
        The method to use. For more information, check the respective papers
        PG16 is preferred over Tremonti04. Tremonti can be used when not all lines for PG16 are present
    SNR: float
        ratio of line flux and error under which measurements should be discarded. Using anything under 2~3
        is not advised. Use the same number as the RemoveBadFit function, or else there may be galaxies without
        a metallicity
    verbose: bool
        if True, prints status messages and additional outputs

    Returns
    -------
    dataframe: Pandas Dataframe
        a column "METAL" has been added to the original dataframe containing the metallicities
    """

    metal_list = []
    del_list = []
    a = 1

    if method == "Tremonti04":      # Tremonti et al (2004), ApJ 613, 898
        
        print("Adding metallcity using Tremonti et al. (2004)...") if verbose else None
        for i, rows in dataframe.iterrows():
            R23 = (float(rows["OIIR_FLUX"]) + float(rows['OIIB_FLUX']) + float(rows["OIIIB_FLUX_x"]) + float(rows["OIIIR_FLUX_x"]))/float(rows["HB_FLUX_y"])

            if R23 < 0:
                del_list.append(i)
            else:
                R23 = np.log10(R23)
                x = 9.185 - 0.313*R23 - 0.264*R23*R23 - 0.321*R23*R23*R23
                # this recipe is fitted in the range 8~9.5. metallicities far outside this range should not be trusted
                if x < 7 or x > 11:
                    del_list.append(i)
                    a +=1
                else:
                    metal_list.append(x)
    

    elif method == "PG16":      # Pilyugin and Grebel (2016), MNRAS 457, 3678

        print("Adding metallicity using Pilyugin and Grebel (2016)...")
        for i, rows in dataframe.iterrows():
            Hbeta = rows["HB_FLUX_y"]
            R3 = np.log10((rows["OIIIR_FLUX_x"] + rows["OIIIB_FLUX_x"])/Hbeta)
            N2 = np.log10((rows["NIIR_FLUX_x"] + rows["NIIB_FLUX_x"])/Hbeta)

            if rows["OII_FITFAIL"] == 1 or rows["OIIR_NPEG"] > 0 or rows["OIIB_NPEG"] > 0 or SNR*rows["OIIR_FLUX_ERR"] > rows["OIIR_FLUX"]:
                S2 = np.log10((rows["SIIR_FLUX"] + rows["SIIB_FLUX"])/Hbeta)
                if N2 >= -0.6:
                    metal = 8.424 + 0.030*(R3-S2) + 0.751*N2 + (-0.349 + 0.182*(R3-S2) + 0.508*N2)*S2
                else:
                    metal = 8.072 + 0.789*(R3-S2) + 0.726*N2 + (1.069 - 0.170*(R3-S2) + 0.022*N2)*S2

            else:
                R2 = np.log10((rows["OIIR_FLUX"] + rows["OIIB_FLUX"])/Hbeta)
                if N2 >= -0.6:
                    metal = 8.589 + 0.022*(R3-R2) + 0.399*N2 + (-0.137 + 0.164*(R3-R2) + 0.589*N2)*R2
                else:
                    metal = 7.932 + 0.944*(R3-R2) + 0.695*N2 + (0.970 - 0.291*(R3-R2) - 0.019*N2)*R2
            # The recipe is fitted in the range 7~9, metallicities far outside this range should not be trusted
            if metal < 6 or metal > 11:
                del_list.append(i)
            else:
                metal_list.append(metal)
            
            
    dataframe = dataframe.drop(del_list)
    metal_list = np.nan_to_num(metal_list, nan=8.5)
    dataframe["METAL"] = metal_list
    print("Added metallicity!") if verbose else None
    print("Galaxies with an abnormal metallicity: {}".format(len(del_list))) if verbose else None

    return dataframe


def RemoveAGN(dataframe, plot=False, verbose=False):
    '''
    Removes galaxies containing an AGN as defined by Kewley et al. (2001), ApJ 556, 121 

    Parameters
    ----------
    dataframe: Pandas Dataframe
        single dataframe consisting of the simple and complex line fits, preferably merged with 
        the pandas.merge function
        HA, HB, OIIIR and SIIB flux need to be positive
    plot: bool
        if true, plots a BPT diagram of SII vs OIII
    verbose: bool
        if True, prints status messages and additional outputs

    Returns
    -------
    dataframe: Pandas dataframe
        Dataframe where all galaxies contaminated by AGN flux are removed
    '''

    print("Removing AGNs...") if verbose else None
    del_list = []

    if plot:
        xline = np.linspace(-1.75, 0.2, 100)
        yline = 0.72/(xline - 0.32) + 1.30
        OIIIratio = []
        SIIratio = []

    for i, rows in dataframe.iterrows():
        OIII = np.log10(rows["OIIIR_FLUX_x"] / rows["HB_FLUX_y"])
        SII = np.log10(rows["SIIB_FLUX"] / rows["HA_FLUX_y"])

        if plot:
            if OIII > -4 and SII > -4:
                OIIIratio.append(np.log10(rows["OIIIR_FLUX_x"] / rows["HB_FLUX_y"]))
                SIIratio.append(np.log10(rows["SIIB_FLUX"] / rows["HA_FLUX_y"]))

        if OIII > 0.72 / (SII - 0.32) + 1.3:
            del_list.append(i)

    dataframe = dataframe.drop(del_list)

    if plot:
        x, y, c = Plots.estimate_density(np.asarray(SIIratio), np.asarray(OIIIratio))

        plt.figure("AGN")
        plt.scatter(x, y, c=c, alpha=0.5)
        #plt.scatter(SIIratio2, OIIIratio2, c=colours[4])
        plt.xlabel(r"$\log([SII]/H\alpha$)")
        plt.ylabel(r"$\log([OIII]/H\beta$)")
        plt.title("")
        plt.plot(xline, yline)
        #plt.show()

    print("Galaxies removed due to spectra contaminated by AGNs: {}".format(len(del_list))) if verbose else None

    return dataframe


def SelectRedshift(dataframe, lower, upper, verbose=False):
    '''
    Remove all objects not in the given redshift interval. if lower is higher than upper, nothing will be returned

    Parameters
    ----------
    dataframe: Pandas Dataframe
        dataframe containing the objects. Needs a column named "Z" with the redshifts
    lower: float
        lower bound for the redshift. Redshifts equal to the bound are included
    upper: float
        upper bound for the redshift. Redshifts equal to the bound are included
    verbose: bool
        if True, prints status messages and additional outputs

    Returns
    -------
    dataframe: Pandas Dataframe
        dataframe with only galaxies between the lower and upper bounds, bounds included

    '''

    print("Removing unwanted redshifts...") if verbose else None
    
    del_list=[]
    for i, rows in dataframe.iterrows():
        if rows["Z"] < lower or rows["Z"] > upper:
            del_list.append(i)

    dataframe = dataframe.drop(del_list)
    print("Removed unwanted redshifts!") if verbose else None
    print("Objects removed: {}".format(len(del_list))) if verbose else None

    return dataframe
    

def AddMass(dataframe, verbose=False):
    '''
    Add a column containing the mass to the dataframe, using the StellarMasses DMU. Not
    every galaxy has the right measurements to add mass, so some data will be lost here
    See the DMU description for more information

    Parameters
    ----------
    dataframe: Pandas Dataframe
        Dataframe created from the StellarMasses DMU
    verbose: bool
        if True, prints status messages and additional outputs

    Returns
    -------
    dataframe: Pandas Dataframe
        the same dataframe as the input, with a column "MASS" added at the end contains the 
        mass of the galaxies in units log10 M*. All columns with insuficient/bad data are removed
    '''

    print("Adding mass...") if verbose else None

    mass_list = []
    del_list = []
    for i, row in dataframe.iterrows():
        # The fluxscale needs to be approx. higher than unity, so an arbitrary 0.8 cut-off is chosen
        if row['fluxscale'] < 0.8:
            del_list.append(i)
        else:
            mass = row['logmstar'] + np.log10(row['fluxscale'])
            mass_list.append(mass)

    dataframe = dataframe.drop(del_list)
    dataframe["MASS"] = mass_list

    print("Added mass!") if verbose else None
    print("Galaxies removed due to a low 'fluxscale' parameter: {}".format(len(del_list))) if verbose else None

    return dataframe


def FluxToMagnitude(dataframe, bands, verbose=False):
    '''
    Originally created to convert flux to magnitude. Every filter system has its own 
    zeropoint flux and this would result in very unpractical code for something that doesn't
    really matter. Now simply takes the log of each element in the columns given by 'bands'
    The results should be the same with and without zeropoint flux: only the ordering 
    itself is important, not the exact value (we use ERF's). And everything gets z-score normalized anyway.

    Parameters
    ----------
    dataframe: Pandas dataframe
        dataframe containing the fluxes
        cannot contain flux values <= 0
    bands: (Numpy) array
        names of the columns which need to be transformed
    verbose: bool
        if True, prints status messages and additional outputs

    Returns
    -------
    dataframe: Pandas Dataframe
        dataframe where the fluxes are replaced by the magnitudes, in place. 
    '''

    print("Taking the log of the flux...") if verbose else None

    X = dataframe.loc[:, bands].values


    for i in range(len(bands)):
        mag = -2.5 * np.log10(X[:, i])
        X[:, i] = mag

    dataframe[bands] = X
    print("Log of flux taken") if verbose else None

    return dataframe


def AddColours(dataframe, bands, neighbours=2, squared=True, verbose=False):
    '''
    Adds colours and optionally squared colours to the dataframe. Colours are defined by difference 
    of the magnitude (or log) of two bands. Same remarks about zeropoint fluxes as the FluxToMagnitude
    function

    Parameters
    ----------
    dataframe: Pandas dataframe
        dataframe containing magnitude values of galaxies
    bands: (Numpy) array
        array containing the names of the columns to calculate the colours
    neighbors: int
        describes how far magnitudes combine to form colours.
        e.g. with neighbours=2, in the sequence a,b,c,d,e: a combines with b and c to form colours but
        no other magnitudes. Only goes one way as a-b and b-a contain the same information
    squared: bool
        if true, also adds squared colours
    verbose: bool
        if True, prints status messages and additional outputs

    Returns
    -------
    dataframe: Pandas dataframe
        original dataframe with the colours added in columns at the end. Names of the columns are colour1-colour2
        and colour1-colour2sq in case squared colours are added
    names: array
        array containing the names of the newly added colours
    '''

    print("Calculating colours...") if verbose else None

    names = []
    X = dataframe.loc[:, bands].values
    colours_arr = np.zeros((np.shape(X)[0], 1))

    for i in range(len(bands)):
        for j in range(i+1, min(len(bands), i + neighbours+1)):
            color = X[:, i] - X[:, j]
            color = np.reshape(color, (-1, 1))
            colorsq = color*color
            names.append(bands[i] + "-" + bands[j])
            colours_arr = np.append(colours_arr, color, axis=1)
            if squared:
                colorsq = np.reshape(colorsq, (-1, 1))
                colours_arr = np.append(colours_arr, colorsq, axis=1)
                names.append(bands[i] + "-" + bands[j] + "sq")

    colours_arr = np.delete(colours_arr, 0, 1)
    dataframe[names] = colours_arr
    print("Added colours!") if verbose else None

    return dataframe, names


def Crossmatch(df1, df2, sep=1.0, verbose=0):
    '''
    Crossmatches the data from 2 databases using astropy
    The function was originally created to merge data from GAMA and the Acquaviva paper 
    and this is reflected in the naming scheme of the internal variables

    Parameters
    ----------
    df1, df2: Pandas dataframes
        dataframes containing the data that has to be merged. Both dataframes need to contain
        a RA and DEC in the same format for this to work
    sep: float
        controls the maximum separation between two objects to be considered the same, in arcseconds
        Higher values find more matches, at the expense of a higher risk of mismatching objects
    verbose: 0, 1, 2
        not a bool this time, so be carefull. 0 is no messages, 1 corresponds to the usual messages when
        this is normally set to true. 2 creates a plot to check what the separation is between matched objects
        and can be helpfull in choosing sep

    Returns
    -------
    dataframe: Pandas dataframe
        a dataframe created by merging df1 and df2, with only objects appearing in both dataframes.
        Obviously the data is matched
    '''

    print("Matching data...") if verbose > 0 else None

    gama_table = Table.from_pandas(df1)
    sdss_table = Table.from_pandas(df2)

    gama_coords = SkyCoord(gama_table['RA']*u.deg, gama_table['DEC']*u.deg)
    sdss_coords = SkyCoord(sdss_table['RA']*u.deg, sdss_table['DEC']*u.deg)

    # Match coordinates and check if it was done correctly
    idx_sdss, d2d_sdss, d3d_sdss = gama_coords.match_to_catalog_sky(sdss_coords)

    # we expect all objects to be within a couple of arcsecs in each catalog
    # if not, we have a problem
    if verbose == 2:
        plt.hist(d2d_sdss.arcsec, histtype='step')
        plt.xlabel('separation (arcsec)')
        plt.show()
    
    max_sep = sep * u.arcsec
    sep_constraint = d2d_sdss < max_sep

    gama_table = gama_table[sep_constraint]
    sdss_table = sdss_table[idx_sdss[sep_constraint]]

    dataframe = pd.merge(gama_table.to_pandas(), sdss_table.to_pandas(), left_index=True, right_index=True)
    
    print("Data matched!") if verbose > 0 else None
    print("Objects which appear in both databases: {}".format(np.shape(sep_constraint)[0])) if verbose > 0 else None

    return dataframe


def FindBadGalaxies(dataframe, verbose=False):
    """
    Function specifically for finding galaxies with at least one fit parameter pegged
    Used to verify suspicion that these are massive ellipticals.
    Looks at lines used by the Pilyugin and Grebel (2016) metallicity estimate

    Parameters
    ----------
    dataframe: Pandas dataframe
        assumed to be the merged GaussFitSimple and GaussFitComplex dataframes from the GAMA website
    verbose: bool
        if True, prints status messages and additional outputs

    Returns
    -------
    Dataframe: Pandas dataframe
        dataframe with only galaxies with parameters pegged left
    """

    print("looking for galaxies with pegged parameters") if verbose else None
    del_list = []

    for i, rows in dataframe.iterrows():
        if rows["OIIIR_NPEG_x"] == 0 and rows["OIIIB_NPEG_x"] == 0 and rows["NIIR_NPEG_x"] == 0 and rows["NIIB_NPEG_x"] == 0 and rows["HB_NPEG_y"] == 0:
            del_list.append(i)

    dataframe = dataframe.drop(del_list)
    print("{} galaxies with pegged parameters left".format(dataframe.shape[0]))

    return dataframe


def PrepCigaleFile(location, verbose=False):
    """
    Reads the CIGALE results and renames the columns to make it work with the rest of the program

    Parameters
    ----------
    location: string
        location of the file
    verbose: bool, default=False
        if True, prints status messages and additional outputs

    Returns
    -------
    cigale: pandas dataframe
    """

    print("Reading the file...") if verbose else None
    cigale = pd.read_csv(location)

    rename_dict = {'id': 'CATAID', 'bayes.galex.FUV': 'FUV_flux', 'bayes.galex.NUV': 'NUV_flux', 'bayes.sdss.up': 'u_flux', 'bayes.sdss.gp': 'g_flux', 'bayes.sdss.rp': 'r_flux', 'bayes.sdss.ip': 'i_flux', 'bayes.sdss.zp': 'z_flux', 'bayes.WFCAM_Z': 'X_flux', 'bayes.vista.vircam.Y': 'Y_flux', 'bayes.vista.vircam.J': 'J_flux', 'bayes.vista.vircam.H': 'H_flux', 'bayes.vista.vircam.Ks': 'K_flux', 'bayes.WISE1': 'W1_flux', 'bayes.WISE2': 'W2_flux', 'bayes.WISE3': 'W3_flux', 'bayes.WISE4': 'W4_flux', 'bayes.herschel.pacs.100': 'P100_flux', 'bayes.herschel.pacs.160': 'P160_flux', 'bayes.PSW': 'S250_flux', 'bayes.PMW': 'S350_flux', 'bayes.PLW': 'S500_flux'}

    cigale = cigale.rename(columns=rename_dict)
    
    print("File read and columns renamed!") if verbose else None
    return cigale