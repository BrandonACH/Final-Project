"""
-----------------------------------------------------------------------------------------------
------------------------------COVID Dataset Our World in Data----------------------------------
-----------------------------------------------------------------------------------------------

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def GetCOVID_Dataset():
    if not os.path.exists('../data/owd/covid19_world.csv'):
    
        # Read OWID COVID-19 dataset
        rawdata = pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv')

        # Convert date columns to datetime type
        rawdata['date'] = pd.to_datetime(rawdata['date'])

        # Filter the DataFrame by date range
        rawdata = rawdata[(rawdata['date'] >= '2020-02-01') & (rawdata['date'] <= '2024-03-31')]

        # Extract list of continents in dataset
        continents = rawdata['continent'].unique()
        continents = [x for x in continents if str(x) != 'nan']
    
        # Initialize an empty list to store DataFrames
        dfs = []        

        # Set an empty DataFrame to further append all countries with processed data
        ColumnsDF=['date','continent','country','iso_code','new_cases','total_cases','new_deaths','total_deaths',
                    'new_vaccinations','people_vaccinated','people_fully_vaccinated',
                'total_boosters','population','population_density','total_vaccinations']
        
        FullData=pd.DataFrame(columns=ColumnsDF)
        
        #-----------Creation of directories to save data-------------------------
        path0 = os.getcwd()
        parent_directory = os.path.dirname(path0)
        data_directory = os.path.join(parent_directory, 'data/owd')
        try:
            os.mkdir(data_directory)
        except OSError:
            print ("Directory %s already exists" % data_directory)
        else:
            print ("Successfully created the directory %s" % data_directory)
        
        # ----------------Iterating over continents i-------------------------
        for i in continents:
            data = rawdata[rawdata['continent'] == i]
            countries = data['location'].unique()
            countries = [country for country in countries if country not in ['Scotland', 'England','Northern Cyprus','Wales','Northern Ireland','Hong Kong','Turkmenistan','North Korea']]
            # ---------------Going inside continents: iterating over countries j -----------
            for j in countries:
                data_i = data[data['location'] == j]
                if len(data_i)>1:
                    time = pd.to_datetime(data_i['date'])
                    iso_code = data_i['iso_code']
                    Total_cases = data_i['total_cases']
                    Total_deaths = data_i['total_deaths']
                    New_cases = data_i['new_cases_smoothed']
                    New_deaths = data_i['new_deaths_smoothed']
                    Population = data_i['population']
                    Density = data_i['population_density']
                    
                    
                    # -----------Filling the gaps of the vaccinated population----------
                    New_vaccinations = data_i['new_vaccinations_smoothed'].replace([0,np.nan], method='ffill')
                    New_vaccinations = New_vaccinations.replace([np.nan], 0)
                    
                    People_fully_vaccinated = data_i['people_fully_vaccinated'].replace([0,np.nan], method='ffill')
                    People_fully_vaccinated = People_fully_vaccinated.replace([np.nan], 0)
                    
                    People_vaccinated = data_i['people_vaccinated'].replace([0,np.nan], method='ffill')
                    People_vaccinated = People_vaccinated.replace([np.nan], 0)
                    
                    Total_boosters = data_i['total_boosters'].replace([0,np.nan], method='ffill')
                    Total_boosters = Total_boosters.replace([np.nan], 0)

                    Total_vaccinations = data_i['total_vaccinations'].replace([0,np.nan], method='ffill')
                    Total_vaccinations = Total_vaccinations.replace([np.nan], 0)
                    
                    Density = Density.replace([np.nan], 0)
                    
                    Total_deaths = Total_deaths.replace([np.nan], 0)
                    Total_cases = Total_cases.replace([np.nan], 0)
                    New_cases = New_cases.replace([np.nan], 0)
                    New_deaths = New_deaths.replace([np.nan], 0)

                    # -----------7-day moving average new COVID cases----------
                    #New_cases = New_cases.rolling(window=7, center=True).mean()

                    # -------------------Saving processed data-----------------------------------
                    DataCountry = {'date': list(time),
                        'iso3': list(iso_code),
                        'new_cases': list(New_cases),
                        'total_cases': list(Total_cases), 
                        'new_deaths': list(New_deaths),
                        'total_deaths': list(Total_deaths), 
                        'new_vaccinations': list(New_vaccinations),
                        'people_vaccinated': list(People_vaccinated),
                        'people_fully_vaccinated': list(People_fully_vaccinated),
                        'total_boosters': list(Total_boosters) ,
                        'population': list(Population) ,
                        'population_density': list(Density),
                        'total_vaccinations': list(Total_vaccinations),
                        }
                    # -----------------Setting processed data into DataFrame & append into global DataFrame----------
                    DataCountry = pd.DataFrame(DataCountry)
                    DataCountry.insert(1, 'continent', i)
                    DataCountry.insert(2, 'country', j)
                    dfs.append(DataCountry)
                    
        # -----------------Return global DataFrame---------------------
        # Concatenate all DataFrames in the list
        FullData = pd.concat(dfs, ignore_index=True)    
        FullData['date']=pd.to_datetime(FullData['date'])
        FullData.loc[FullData['iso3'] == "OWID_KOS", 'iso3'] = 'XKX'
        # Save the processed DataFrame as CSV
        filename = os.path.join(data_directory, 'covid19_world.csv')
        FullData.to_csv(filename, index=False)  # Save DataFrame to CSV without index


"""
-----------------------------------------------------------------------------------------------
------------------------------COVID Variants GISAID--------------------------------------------
-----------------------------------------------------------------------------------------------

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

def GetVariants():
    if not os.path.exists('../data/gisaid/variants.csv'):
    
        # Read GISAID Statistics
        df = pd.read_csv("../data/gisaid/gisaid_variants_statistics.tsv", sep='\t')

        # Convert date columns to datetime type
        df['Week prior to'] = pd.to_datetime(df['Week prior to'])

        # Filter the DataFrame by date range
        df = df[(df['Week prior to'] >= '2020-02-01') & (df['Week prior to'] <= '2024-03-31')]

        # Filter DataFrame to show only COVID Variants
        df = df[df['Type'].isin(['Variant'])]

        # Only keep BA. Omicron Lineages and original Omicron lineage B.1.1.529, all others will be worked only with their name 
        #df = df[~((df['Type'] == 'Lineage') & (~df['Value'].str.startswith('BA.') | ~df['Value'].str.contains('B.1.1.529')))]

        # # Extracting only one number after the dot for BA.#
        # mask = df['Type'] == 'Lineage'
        # df.loc[mask, 'Value'] = df.loc[mask, 'Value'].str.split('.').str[0] + '.' + df.loc[mask, 'Value'].str.split('.').str[1].str[0]

        # Add 'Omicron' to corresponding lineages
        mask = df['Value'].str.startswith('VOI') | df['Value'].str.startswith('VUM') | df['Value'].str.startswith('VOC')
        df.loc[mask, 'Value'] = 'Omicron ' + '(' + df.loc[mask, 'Value'].str.extract(r'\((.*?)\+?\)', expand=False) + ')'

        # IHU Variant ()
        mask = df['Value'].str.contains('B.1.640')
        df.loc[mask, 'Value'] = 'IHU ' + '(' + df.loc[mask, 'Value'].str.extract(r'\((.*?)\+?\)', expand=False) + ')'

        # Extracting Alpha, Beta, Gamma, Delta from Value
        pattern = r'(Alpha|Beta|Gamma|Delta|Epsilon|Eta|Iota|Kappa|Lambda|Mu|Theta|Omicron|Zeta)'
        mask = (df['Type'] == 'Variant') & df['Value'].str.contains(pattern, regex=True)
        df.loc[mask, 'Value'] = df.loc[mask, 'Value'].str.extract(pattern, expand=False) + ' (' + df.loc[mask, 'Value'].str.extract(r'\((.*?)\+?\)', expand=False) + ')'

        # Set Omicron (XBB+XBB.* excluding XBB.1.5, XBB.1.16, XBB.1.9.1, XBB.1.9.2, XBB.2.3) to Omicron(XBB) for easier reading
        mask = df['Value'] == 'Omicron (XBB+XBB.* excluding XBB.1.5, XBB.1.16, XBB.1.9.1, XBB.1.9.2, XBB.2.3)'
        df.loc[mask, 'Value'] = 'Omicron ' + '(XBB)'

        # # Delete Former VOC Omicron GRA (B.1.1.529+BA.*) first detected in Botswana/Hong Kong/South Africa, as we already have lineages
        # df = df[~(df['Value'] == 'Former VOC Omicron GRA (B.1.1.529+BA.*) first detected in Botswana/Hong Kong/South Africa')] 
        
        # #Grouping B.1.640 into Other group
        # mask = df['Value'].str.contains('B.1.640', regex=False)
        # df.loc[mask, 'Value'] = 'Other'

        # # Extracting Omicron subvariants
        # pattern = r'\((.*?)\+'
        # mask = (df['Type'] == 'Variant') & df['Value'].str.contains(pattern, regex=True)
        # df.loc[mask, 'Value'] = 'Omicron ' + '(' + df.loc[mask,'Value'].str.extract(pattern, expand=False) + ')'

        # # Grouping by Country, Week prior to, and Value, and aggregating Submission Count, '% per Country and Week', and 'Total per Country and Week'
        # df = df.groupby(['Country', 'Week prior to', 'Value']).agg({
        # 'Submission Count': 'sum',
        # '% per Country and Week': 'sum',
        # 'Total per Country and Week': 'sum'
        # }).reset_index()

        df.rename(columns={'Country':'country'},inplace=True)
        
        #Set working directory
        path0 = os.getcwd()
        parent_directory = os.path.dirname(path0)
        data_directory = os.path.join(parent_directory, 'data/gisaid')

        #Save CSV File
        filename = os.path.join(data_directory, 'variants.csv')
        df.to_csv(filename, index=False)  # Save DataFrame to CSV without index



"""
-----------------------------------------------------------------------------------------------
------------------------------Climate CDS API-------------------------------------------------
-----------------------------------------------------------------------------------------------

"""
import cdsapi
import xarray as xr
from shapely.geometry import Point, MultiPolygon
from tqdm import tqdm
from metpy.calc import relative_humidity_from_dewpoint

class ClimateDataProcessor:
    def __init__(self):
        self.country_to_iso = None
        self.world = None

    def retrieve_climate_data(self):
        # Data retrieval using CDS API

        c = cdsapi.Client()
        year_months = {'2020': [str(i).zfill(2) for i in range(2, 13)],
                    '2021': [str(i).zfill(2) for i in range(1, 13)],
                    '2022': [str(i).zfill(2) for i in range(1, 13)],
                    '2023': [str(i).zfill(2) for i in range(1, 13)],
                    '2024': [str(i).zfill(2) for i in range(1, 4)]}

        for year, months in year_months.items():
            for month in months:
                for i, days in enumerate([list(range(1, 11)), list(range(11, 21)), list(range(21, 32))]):
                    if not os.path.exists(f'../data/climate/{year}_{month}_{i}_weather.nc'):
                        c.retrieve(
                            'reanalysis-era5-single-levels',
                            {
                                'product_type': 'reanalysis',
                                'format': 'netcdf', 
                                'grid': [.5, .5],
                                'variable': [
                                    '2m_dewpoint_temperature',
                                    '2m_temperature',
                                    'mean_sea_level_pressure',
                                    'total_precipitation',
                                ],
                                'year': year,
                                'month': [month],
                                'day': [str(d).zfill(2) for d in days],
                                'time': [
                                    '00:00', '01:00', '02:00',
                                    '03:00', '04:00', '05:00',
                                    '06:00', '07:00', '08:00',
                                    '09:00', '10:00', '11:00',
                                    '12:00', '13:00', '14:00',
                                    '15:00', '16:00', '17:00',
                                    '18:00', '19:00', '20:00',
                                    '21:00', '22:00', '23:00',
                                ],

                            },
                            f'../data/climate/{year}_{month}_{i}_weather.nc')

    def filter_relevant_coordinates(self):
    
        if not os.path.exists('../data/coords_region.csv'):
            lon_lat = (xr.load_dataset('../data/climate/2023_01_0_weather.nc')
                    .to_dataframe()
                    .reset_index()
                    [['longitude', 'latitude']]
                    .drop_duplicates()
                    .assign(longitude=lambda dd: dd.longitude.apply(lambda x: x if x < 180 else x - 360))
                    )
        self.world = (gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')) 
                .rename(columns={'iso_a3': 'iso3', 'name': 'country'})
                .loc[lambda dd:  ~dd.country.isin(['Antarctica'])]
                )
        # Extract the original geometry of France
        original_geometry = self.world.loc[self.world['country'] == 'France', 'geometry'].values[0]

        # Create a new MultiPolygon excluding the first polygon
        new_polygons = list(original_geometry.geoms)[1:]  # Extract all polygons except the first one
        new_multi_polygon = MultiPolygon(new_polygons)

        # Update the geometry of France in the world GeoDataFrame
        self.world.loc[self.world['country'] == 'France', 'geometry'] = new_multi_polygon

        # Change predefined iso codes to standarized ISO3 codes
        self.world.loc[lambda dd: dd.country=='France', 'iso3'] = 'FRA'
        self.world.loc[lambda dd: dd.country=='Norway', 'iso3'] = 'NOR'
        self.world.loc[lambda dd: dd.country=='Kosovo', 'iso3'] = 'XKX'

        # Assign regions to pairs of longitude and latitudes
        self.country_to_iso = self.world[['country', 'iso3']]
        if not os.path.exists('../data/coords_region.csv'):
            lon_lat_region = (lon_lat
                            .drop_duplicates()
                            .assign(country=lambda dd: [region_assign(lon, lat, shape_df=self.world,
                                                                        col_name='country', res=.5)
                                                        for lon, lat 
                                                        in tqdm(zip(dd.longitude, dd.latitude),
                                                                leave=False, total=dd.shape[0])])
                            .dropna()
            )
            lon_lat_region.to_csv('../data/coords_region.csv', index=False)
        

    def compute_relative_absolute_humidity(self):
        lon_lat_region = pd.read_csv('../data/coords_region.csv').merge(self.country_to_iso)

        # Compute relative and absolute humidity
        nc_files = [f for f in os.listdir('../data/climate/') if f.endswith('.nc')]

        for file in tqdm(nc_files, leave=False, desc='Processing NC file'):
            if not os.path.exists(f'../data/climate/processed/{file.split(".")[0]}.pickle'):
                xr_file = xr.load_dataset(f'../data/climate/{file}')
                xr_file.d2m.attrs['units'] = 'degK'
                xr_file.t2m.attrs['units'] = 'degK'
                xr_file.msl.attrs['units'] = 'Pa'
                xr_file.tp.attrs['units'] = 'm'
                (xr_file
                    .assign(rh=lambda dd: relative_humidity_from_dewpoint(dd.t2m, dd.d2m))
                    .resample(time='1D')
                    .mean()
                    .metpy.dequantify()
                    .to_dataframe()
                    .reset_index()
                    .assign(longitude=lambda dd: dd.longitude.apply(lambda x: x if x < 180 else x - 360))
                    .merge(lon_lat_region)
                    .dropna()
                    .assign(tp=lambda dd: dd.tp * 24 * 1000)
                    .assign(ah=lambda dd: calculate_absolute_humidity(dd['t2m'], dd['msl'], dd['rh']))
                    .rename(columns={'t2m': 'temperature',
                                    'tp': 'total_precipitation',
                                    'rh': 'relative_humidity',
                                    'ah': 'absolute_humidity',
                                    'time': 'date'})
                    .assign(temperature=lambda dd: dd.temperature - 273.15)
                    [['date', 'latitude', 'longitude', 'country', 'temperature','absolute_humidity', 'relative_humidity', 'total_precipitation']]
                    .to_pickle(f'../data/climate/processed/{file.split(".")[0]}.pickle')
                )

    def concat_data(self):
        processed_path = "../data/climate/processed/"
        relevant_files = [f for f in os.listdir(processed_path)]
        climate_df = pd.concat([pd.read_pickle(f'{processed_path}{f}') for f in relevant_files])
        climate_df = climate_df.merge(self.country_to_iso,on='country')
        return climate_df, self.world, self.country_to_iso

   


"""
-----------------------------------------------------------------------------------------------
------------------------------Socioeconomical Variables--------------------------------------------
-----------------------------------------------------------------------------------------------

"""
import wbdata
import pandas as pd

def GetSocioEco():
    # Check if the data file exists
    if not os.path.exists('../data/worldbank/socioeconomical.csv'):
        # Get country codes and names from the World Bank Repository
        code_country_wb = wbdata.get_countries()
        code_country_wb = pd.DataFrame(code_country_wb)[['id', 'name']]

        # Get ISO3 country codes and names from OWID
        covid_df = pd.read_csv('../data/owd/covid19_world.csv')
        code_country_iso3 = covid_df[['iso3', 'country']].drop_duplicates()

        # Merge ISO3 and ID from World Bank Repo
        wb_to_iso3 = pd.merge(code_country_iso3, code_country_wb, left_on='iso3', right_on='id')

        # Define the indicators to fetch
        indicators = {
            'EN.POP.DNST': 'Population density',
            "NY.GDP.PCAP.CD": "GDP per capita",
            'NY.GDP.DEFL.KD.ZG': 'Inflation rate',
            "SH.XPD.CHEX.PC.CD": 'Health expenditure per capita',
            "SL.UEM.TOTL.NE.ZS": "Unemployment rate",
            "SI.POV.DDAY": "Extreme poverty",
            "SP.DYN.LE00.IN": "Life expectancy",
            "SP.DYN.CDRT.IN": 'Crude death rate',
            "SP.POP.65UP.TO.ZS": 'Population aged 65 and above',
            'SH.PRV.SMOK': 'Tobacco use',
            'SH.MED.BEDS.ZS': 'Hospital beds',
            'IS.AIR.PSGR': 'Air Passengers',
            'SE.SEC.CUAT.LO.ZS': 'Lower secondary',
            'SE.PRM.ENRL.TC.ZS': 'Pupil-teacher ratio',
            'SE.XPD.TOTL.GD.ZS': 'Education expenditure',
            'SH.MED.NUMW.P3': 'Nurses and Midwives',
            'SH.MED.PHYS.ZS': 'Physicians',
            'SH.STA.DIAB.ZS': 'Diabetes prevalence'
        }

        # Fetch data
        countries = wb_to_iso3['iso3']
        dfa = wbdata.get_dataframe(indicators, country=countries, parse_dates=True)

        # Filter data for the most recent year available for each country, considering the last 10 years
        dfa_filtered = dfa.sort_index().loc(axis=0)[:, '2010-01-01':'2020-01-01'].groupby('country').fillna(method='bfill', limit=1).fillna(method='ffill')
        dfa = dfa_filtered.loc(axis=0)[:, '2019-01-01'].reset_index(level='date').iloc[:, 1:]

        # Merge with ISO3 country codes
        socioeconomic = pd.merge(dfa, wb_to_iso3, left_on='country', right_on='name', how='left')
        socioeconomic.drop(columns=['id', 'name', 'country_x'], inplace=True)
        socioeconomic.rename(columns={'country_y': 'country'}, inplace=True)

        # Rearrange columns
        socioeconomic = socioeconomic[socioeconomic.columns[-2:].tolist() + socioeconomic.columns[:-2].tolist()]

        # Sum Nurses and Midwives with Physicians to create Health Personnel column
        socioeconomic['Health personnel'] = socioeconomic['Nurses and Midwives'] + socioeconomic['Physicians']
        socioeconomic.drop(columns=['Nurses and Midwives', 'Physicians'], inplace=True)

        # Set working directory
        data_directory = os.path.join(os.path.dirname(os.getcwd()), 'data', 'worldbank')

        # Save CSV File
        filename = os.path.join(data_directory, 'socioeconomical.csv')
        socioeconomic.to_csv(filename, index=False)

    

"""
-----------------------------------------------------------------------------------------------
------------------------------Assign regions by latitude and longitude-------------------------
-----------------------------------------------------------------------------------------------

"""

from shapely.geometry import box
import geopandas as gpd

def region_assign(lon: float, lat: float, shape_df: gpd.GeoDataFrame,
                  col_name: str, res: float) -> str:
    """

    Parameters
    ----------
    lon
        Longitudinal coordinate in the geometry of shape_df
    lat
        Latitudinal coordinate in the geometry of shape_df
    shape_df
        GeoPandas DataFrame containing an assigned region for each exclusive geometry
    col_name
        Column name
    res

    Returns
    -------

    """

    cell_box = box(lon - res / 2, lat - res / 2, lon + res / 2, lat + res / 2)

    intersecting_regions = (shape_df
                           .loc[lambda dd: dd.geometry.apply(lambda x: cell_box.intersects(x))]
        )
    if intersecting_regions.shape[0] == 0:
        return np.NaN
    elif intersecting_regions.shape[0] == 1:
        return intersecting_regions[col_name].values[0]
    else:
        intersecting_regions = (
            intersecting_regions
                .assign(area=lambda dd: dd.geometry.apply(lambda x: x.intersection(cell_box).area))
                .loc[lambda dd: dd['area'] == dd['area'].max()])
        return intersecting_regions[col_name].values[0]


"""
-----------------------------------------------------------------------------------------------
------------------------------Calculation of Absolute Humidity---------------------------------
-----------------------------------------------------------------------------------------------

"""
import numpy as np

# How to calculate Absolute Humidity (https://github.com/atmos-python/atmos/issues/3)
def calculate_absolute_humidity(t2m, msl, RH):
    es = 611.2*np.exp(17.67*(t2m-273.15)/(t2m-29.65))
    rvs = 0.622*es/(msl - es)
    rv = RH * rvs
    qv = rv/(1 + rv)
    rho = msl/(287.*t2m)
    AH = qv*rho
    return AH*1000