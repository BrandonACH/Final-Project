
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
        #rawdata = rawdata[(rawdata['date'] >= start) & (rawdata['date'] <= end)]

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
        data_directory = os.path.join(parent_directory, 'data/covid')
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
                    New_cases = data_i['new_cases']
                    New_deaths = data_i['new_deaths']
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

                    # -----------7-day moving average new COVID cases (2 times)----------
                    New_cases = New_cases.rolling(window=7, center=True).mean().rolling(window=7, center=True).mean()
                    New_deaths = New_deaths.rolling(window=7, center=True).mean().rolling(window=7, center=True).mean()

                    Total_cases = New_cases.cumsum()
                    Total_deaths = New_deaths.cumsum()
                    

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

        # Delete first and last week of data, lost during smoothing
        min_date = FullData['date'].min()
        max_date = FullData['date'].max()

        start_of_first_week = min_date + pd.DateOffset(weeks=1)
        end_of_last_week = max_date - pd.DateOffset(weeks=1)

        FullData = FullData[(FullData['date'] >= start_of_first_week) & (FullData['date'] <= end_of_last_week)]

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

        # Filter DataFrame to show only COVID Variants
        df = df[df['Type'].isin(['Variant'])]

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

        df.rename(columns={'Country':'country'},inplace=True)
        df.loc[df['country'] == 'USA', 'country'] = 'United States of America'
        df.loc[df['country'] == "Cote d'Ivoire", 'country'] = "Côte d'Ivoire"
        df.loc[df['country'] == "South Sudan", 'country'] = "S. Sudan"
        df.loc[df['country'] == "Democratic Republic of the Congo", 'country'] = 'Dem. Rep. Congo'
        df.loc[df['country'] == "Czech Republic", 'country'] = 'Czechia'
        df.loc[df['country'] == "Bosnia and Herzegovina", 'country'] = 'Bosnia and Herz.'
        df.loc[df['country'] == "Central African Republic", 'country'] = 'Central African Rep.'
        df.loc[df['country'] == "Republic of the Congo", 'country'] = 'Congo'

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
        """
        Retrieves climate data from the CDS API and saves it in netCDF format.
        Args:
        - grid_size: The grid size for the climate data.
        """

        c = cdsapi.Client()
        year_months = {'2021': [str(i).zfill(2) for i in range(1,13)],
                       '2022': [str(i).zfill(2) for i in range(1,13)],
                       '2023': [str(i).zfill(2) for i in range(1,2)]
                      }
        
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
        """
        Filters relevant coordinates and assigns regions to pairs of longitude and latitudes.
        """
        
        # Load a sample netCDF file to get unique longitude and latitude values
        if not os.path.exists('../data/coords_region.csv'):
            lon_lat = (
                xr.load_dataset('../data/climate/2022_01_0_weather.nc')
                .to_dataframe()
                .reset_index()
                [['longitude', 'latitude']]
                .drop_duplicates()
                .assign(longitude=lambda dd: dd.longitude.apply(lambda x: x if x < 180 else x - 360))
            )

        # Load world map excluding Antarctica and fix specific ISO codes
        self.world = (gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')) 
                .rename(columns={'iso_a3': 'iso3', 'name': 'country'})
                .loc[lambda dd:  ~dd.country.isin(['Antarctica'])]
                )
        
        # Fix the geometry for France by excluding the first polygon
        original_geometry = self.world.loc[self.world['country'] == 'France', 'geometry'].values[0]
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
        """
        Computes relative and absolute humidity and saves the processed data.
        """

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
        """
        Concatenates processed climate data files into a single DataFrame.
        Returns:
        - climate_df (DataFrame): Concatenated climate data.
        - world (GeoDataFrame): World GeoDataFrame with country geometries.
        - country_to_iso (DataFrame): DataFrame mapping countries to ISO3 codes.
        """

        processed_path = f"../data/climate/processed/"
        relevant_files = [f for f in os.listdir(processed_path)]
        climate_df = pd.concat([pd.read_pickle(f'{processed_path}{f}') for f in relevant_files])
        climate_df = climate_df.merge(self.country_to_iso,on='country')
        return climate_df, self.world, self.country_to_iso

   
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
    """
    Computes the absolute humidity of the air given the temperature (t2m), mean sea level pressure (msl), and relative humidity (RH)
    """
    es = 611.2*np.exp(17.67*(t2m-273.15)/(t2m-29.65))
    rvs = 0.622*es/(msl - es)
    rv = RH * rvs
    qv = rv/(1 + rv)
    rho = msl/(287.*t2m)
    AH = qv*rho
    return AH*1000