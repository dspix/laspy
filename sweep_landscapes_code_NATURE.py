import os
from datetime import datetime
import datetime as dt

from pyproj import Transformer

import matplotlib.pyplot as plt
import ee
import numpy as np
import pandas as pd

from scipy.stats import t as tdstr
import math

import ssqueezepy as ssq
from ssqueezepy.experimental import scale_to_freq
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import fnmatch

from dlm_functions import run_dlm
from helper_functions import calculate_bounds, to_df, get_api_response, wave_variance
from sweep_landscape_functions import sweep_locate, sweep_infil, sweep_extract, plot_landscape, plot_landscape_gif
from gee_import import Mod09gq_profiler, Mod13Q1_profiler

# Change the current working directory to the specified path
os.chdir("C:/Users/Will.Rust/OneDrive - Cranfield University/postdoc/Environment/Projects/RESTRECO/sweep_paper")

#set locations to save cosmos and modis data
cosmos_data_path = "data_in/cosmos_daily/"
cosmos_gcc_path = "data_in/cosmos_gcc/"
modis_daily_path = "data_in/mod09gq/"
modis_comp_path = "data_in/mod13q1/"

#download cosmos data using API
if not os.path.exists(cosmos_data_path):
    os.makedirs(cosmos_data_path)

# Base URL for the COSMOS-UK API
BASE_URL = 'https://cosmos-api.ceh.ac.uk'

# Step 1: Get the site list and store it in a DataFrame
site_info_url = f'{BASE_URL}/collections/1D/locations'
site_info_response = get_api_response(site_info_url)

site_info = {}
for site in site_info_response['features']:
    site_id = site['id']
    site_name = site['properties']['label']
    coordinates = site['geometry']['coordinates']
    date_range = site['properties']['datetime']
    start_date, end_date = date_range.split('/')
    other_info = site['properties']['siteInfo']
    other_info = {key: d['value'] for key, d in other_info.items()}

    site_info[site_id] = {
        'site_name': site_name,
        'coordinates': coordinates,
        'start_date': start_date,
        'end_date': end_date,
        **other_info
    }

cosmos_sites_df = pd.DataFrame.from_dict(site_info, orient='index')

#DOWNLOAD DAILY VWC + GCC DATA
overwrite = False #True

def format_datetime(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

start_date = format_datetime(datetime(2010, 1, 1))
end_date = format_datetime(datetime(2024, 8, 15))
query_date_range = f'{start_date}/{end_date}'

for site_id in cosmos_sites_df.index:
    path = f'{cosmos_data_path}{site_id}_tsdata.csv'

    for root, dirnames, filenames in os.walk(cosmos_gcc_path):
        for filename in filenames:
            if fnmatch.fnmatch(filename, f'*{site_id}*.csv'):
                gcc_path = os.path.join(root, filename)

    if os.path.exists(path) and overwrite != True:
        continue

    data_url = f'{BASE_URL}/collections/1D/locations/{site_id}?parameter-name=cosmos_vwc,precip&datetime={query_date_range}'
    cosmos_vwc_data = get_api_response(data_url)

    coverage = cosmos_vwc_data['coverages'][0]
    identifier = coverage['dct:identifier']
    t_values = coverage['domain']['axes']['t']['values']
    z_values = coverage['ranges']['cosmos_vwc']['values']
    p_values = coverage['ranges']['precip']['values']

    if gcc_path is None:

        df_api = pd.DataFrame({
            'dt': t_values,
            'cosmos_vwc': z_values,
            'precip': p_values,
            'GCC': np.nan
        })

    else:

        df_api = pd.DataFrame({
        'dt': t_values,
        'cosmos_vwc': z_values,
        'precip': p_values

        })  

        cosmos_gcc_df = pd.read_csv(gcc_path, parse_dates=['DATE_TIME'])
        cosmos_gcc_df.replace(-9999, np.nan, inplace=True)

        cosmos_gcc_df = cosmos_gcc_df[['DATE_TIME', 'GCC']]
        cosmos_gcc_df = cosmos_gcc_df.rename(columns={'DATE_TIME': 'dt'})

        df_api['dt'] = pd.to_datetime(df_api['dt']).dt.date
        cosmos_gcc_df['dt'] = pd.to_datetime(cosmos_gcc_df['dt']).dt.date

        df_api = df_api.merge(cosmos_gcc_df, how='left', on='dt')

    print(f'Data for site {site_id} downloaded.')
    df_api.to_csv(path, index=False)

#####DOWNLOAD MODIS DATA#####
if not os.path.exists(modis_daily_path):
    os.makedirs(modis_daily_path)

if not os.path.exists(modis_comp_path):
    os.makedirs(modis_comp_path)

ee.Authenticate()
ee.Initialize()

profiler = Mod09gq_profiler()

for index, row in cosmos_sites_df.iterrows():

    latlon = row['coordinates']
    lat = latlon[0]
    lon = latlon[1]
    name = index
    path_row = modis_daily_path + name + "/" 

    if os.path.exists(path_row):
        continue
    else: 
        os.makedirs(path_row)

    profiles={}
    point=ee.Geometry.Point(lon, lat)          # this is where your location goes
    for y in range(2001,2023):                      # range of years goes here (its easier to trim afterwards)
        sd = dt.datetime(y,1,1).isoformat()[:10]
        ed = dt.datetime(y,12,31).isoformat()[:10]

        p = profiler.profile(point, sd, ed)
        profiles[y] = to_df(p)

    for key in profiles:
        profiles[key].to_csv(path_row +  str(key)+".csv")

    print(name)

#DOWNLOAD COMPOSITE MODIS
profiler = Mod13Q1_profiler()

for index, row in cosmos_sites_df.iterrows():

    latlon = row['coordinates']
    lat = latlon[0]
    lon = latlon[1]
    name = index
    path_row = modis_comp_path + name + "/"

    if os.path.exists(path_row):
        continue
    else: 
        os.makedirs(path_row)

    profiles={}
    point=ee.Geometry.Point(lon, lat)               # this is where your location goes
    for y in range(2001,2023):                      # range of years goes here (its easier to trim afterwards)
        sd = dt.datetime(y,1,1).isoformat()[:10]
        ed = dt.datetime(y,12,31).isoformat()[:10]

        p = profiler.profile(point, sd, ed)
        profiles[y] = to_df(p)


    for key in profiles:
        profiles[key].to_csv(path_row +  str(key)+".csv")

    print(name)


cosmos_data = {}

for site, row in cosmos_sites_df.iterrows():
    cosmos_daily = pd.read_csv(f'{cosmos_data_path}{site}_tsdata.csv', parse_dates=['dt'])

    # daily modis
    modis_files = sorted(os.listdir(os.path.join(modis_daily_path, site)))
    modis_files = [os.path.join(modis_daily_path, site, f) for f in modis_files]

    daily_ndvi = pd.concat([pd.read_csv(f) for f in modis_files])
    daily_ndvi["dt"] = pd.to_datetime(daily_ndvi["date"])

    # daily composite modis
    comp_files = sorted(os.listdir(os.path.join(modis_comp_path, site)))
    comp_files = [os.path.join(modis_comp_path, site, f) for f in comp_files]

    comp_ndvi = pd.concat([pd.read_csv(f) for f in comp_files])
    comp_ndvi["dt"] = pd.to_datetime(comp_ndvi["date"])

    # merge all files
    data_compare = pd.merge(
        daily_ndvi[['dt', "ndvi"]],
        comp_ndvi[['dt', "ndvi"]],
        on="dt",
        how="left",
        suffixes=("_d", "_c"),
    )

    # print(cosmos_daily.columns)
    data_compare = pd.merge(data_compare, cosmos_daily, on='dt', how="inner")
    cosmos_data[site] = data_compare
    print(site)


#generate map of final site locations
exclude_list = ['EUSTN', 'HENFS', 'HILLB', 'COCHN',  'WYTH1', 'REDHL', 'HARWD', 'CGARW', 'ALIC1'] #check bickly

df = cosmos_sites_df[cosmos_sites_df['land_cover'] != 'Arable and horticulture']
df = df[~df.index.isin(exclude_list)]

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from pyproj import Transformer
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Transformer

# Assuming your DataFrame is named df
# Convert the 'coordinates' list into separate latitude and longitude columns
df['latitude'] = df['coordinates'].apply(lambda x: x[0])
df['longitude'] = df['coordinates'].apply(lambda x: x[1])

transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
df['x'], df['y'] = transformer.transform(df['longitude'].values, df['latitude'].values)


# Create a figure with a specific projection (PlateCarree for global)
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.OSGB()})

# Add country boundaries and coastlines
ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1)
ax.add_feature(cfeature.COASTLINE)

# Set extent to focus on the country of interest (e.g., United Kingdom)
ax.set_extent([-10, 2, 49, 61])  # Adjust for other countries

# Plot the points from your DataFrame
ax.scatter(df['x'], df['y'], color='red', s=50, edgecolor='black', transform=ccrs.OSGB(), label='COSMOS Site')

ax.set_xlabel('Easting (m)', fontsize=12)
ax.set_ylabel('Northing (m)', fontsize=12)

# Add a grid for better readability of x and y coordinates
ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)

# Add legend and title
fig.legend()

path_name = f"data_out/figs/map.png"
fig.savefig(path_name, dpi=300, bbox_inches='tight', pad_inches=0)


#SWEEEP and DLM parameters
nseas = [1, 2]
rseas = nseas
wav = ('morlet', {'mu': 6})
fs = 1
vid = 2

l_period =  128 #365.25 / 2
u_period = 365.25 * 1.5
min_thresh = 0.1# 0.2
fs = 1  #sample rate
buff = 0.1
sig_lvl = 0.05
cost = 0.5
cost_i = 0
quantile = 0.9      #quantile used for CSD detection

exp = 1
mean_step = 0.05
range_step = mean_step/2

win_mean = np.arange(mean_step, 1 + mean_step, mean_step)
win_range = np.arange(range_step, 1 + range_step, range_step)

n = 1000

#calculating significant takes a while, have commended out and hard coded calcualted sig value. If data is updated this needs to be recalculated

#sorted_synth_wp = synth_max_wp(win_mean, win_range, wav, fs, n, l_period, u_period)
#sig_ind = int(np.ceil((1 - sig_lvl) * n)) - 1
#sig_wp = sorted_synth_wp[sig_ind]

sig_wp = 0.13964520394802094

sweep_results_table = pd.DataFrame(columns = [
        'Site',     
        'comp int',                     
        'comp cpr',
        'sweep int',
        'sweep cpr'])

dlm_results_table = pd.DataFrame(columns=['Site', 
           'comp RMSE', 
           'comp R2', 
           'comp ar1 var',
           'sweep RMSE', 
           'sweep R2', 
           'sweep ar1 var'])

#provide a warmup (days) where validation and surfaces are not calculated
warmup = int(365 * 1.5)

#import surface plot parameters
surface_plot_params = pd.read_csv('COSMOS_3d_plot_params.csv')

df.to_csv('sweep_sites.csv', index=False)

for i in range(len(cosmos_data)):
#for i in sites_i:
    print(round(i/len(cosmos_data), 2))

    site_data = cosmos_sites_df.iloc[i]
    plot_title = f"Site: {site_data['site_name']}, Land use: {site_data['land_cover']}"
    code = cosmos_sites_df.index[i]

    if site_data['land_cover'] == 'Arable and horticulture':
        continue

    if code in exclude_list:
        continue

    cosmos_df = cosmos_data[code]

    x = cosmos_df["ndvi_d"]
    x = x.values.ravel()

    signal_index, signal_location, best_power, sig_vector, av_power_mat, power_mat, center_mat, range_mat, center_mat_w, range_mat_w = sweep_locate(x, win_mean, win_range, wav, fs, l_period, u_period, min_thresh, sig_wp, cost_i, buff)
    signal_index_infil, signal_index_sigonly, signal_location_infil, signal_location_sigonly = sweep_infil(signal_index, signal_location, best_power, sig_vector, av_power_mat, win_mean, win_range, buff)

    loc_mean_if = signal_location_infil['loc_center']
    loc_max_if = signal_location_infil['loc_max']
    loc_min_if = signal_location_infil['loc_min']

    best_center_ind_if = signal_index_infil['ind_center']
    best_center_ind_sig = signal_index_sigonly['ind_center']

    #SWEEP EXTRATIONS
    x_sweep, wx_og, wx_filt, factor_mask = sweep_extract(x, signal_location_infil, wav, fs, l_period, exp)
    cosmos_df['x_sweep'] = x_sweep

    x_sweep_sig = x_sweep.copy()
    x_sweep_nonsig = x_sweep.copy()

    x_sweep_sig[~sig_vector] = np.nan
    x_sweep_nonsig[sig_vector] = np.nan
    
    ########################
    #DLM - PREPROCESSING
    ########################

    ndvi_comp = cosmos_df[['dt', 'ndvi_c']]
    mask = ~np.isnan(ndvi_comp['ndvi_c'])
    ndvi_comp = ndvi_comp[mask]
    ndvi_comp = ndvi_comp.reset_index(drop = True)

    #identify missing dates in GCC
    missing_dates = cosmos_df[cosmos_df['GCC'].isna()]['dt'].tolist()
    sweep_rm = cosmos_df["dt"].isin(missing_dates)
    comp_rm = ndvi_comp["dt"].isin(missing_dates)
    
    #define 1step and 16step delta
    fs1_deltas = np.ones(4) * 0.996
    fs16_deltas = np.ones(4) * 0.98

    #calculate climate anomalies for daily
    cosmos_df['dt'] = pd.to_datetime(cosmos_df['dt'])
    cosmos_df['month'] = cosmos_df['dt'].dt.month
    monthly_mean_precip = cosmos_df.groupby(['month'])['precip'].transform('mean')
    cosmos_df['precip_an'] = cosmos_df['precip'] - monthly_mean_precip
    cosmos_df = cosmos_df.fillna({'precip_an':0})
    anCLM = cosmos_df['precip_an']

    #calculate climate anomalies for composite
    ndvi_comp = cosmos_df[['dt', 'ndvi_c']]
    mask = ~np.isnan(ndvi_comp['ndvi_c'])
    ndvi_comp = ndvi_comp[mask]
    ndvi_comp = ndvi_comp.reset_index(drop = True)
    anCLM_c = anCLM[mask]
    anCLM_c = anCLM_c.reset_index(drop = True)

    ndvi_comp = pd.merge(ndvi_comp, cosmos_df[['dt', 'GCC']], on='dt', how='left')

    #remove composite outliers
    std_thres = 1.5
    ndvi_comp['month'] = ndvi_comp['dt'].dt.month
    composite_std = cosmos_df.groupby(['month'])['ndvi_c'].agg(['mean', 'std']).reset_index()
    ndvi_comp = pd.merge(ndvi_comp, composite_std, on='month')

    # Calculate the upper and lower bounds
    ndvi_comp['lower_bound'] = ndvi_comp['mean'] - (std_thres * ndvi_comp['std'])
    ndvi_comp['upper_bound'] = ndvi_comp['mean'] + (std_thres * ndvi_comp['std'])

    # Filter the DataFrame to keep values within the bounds
    ndvi_comp['ndvi_c_filt'] = np.where((ndvi_comp['ndvi_c'] < ndvi_comp['lower_bound']) | (ndvi_comp['ndvi_c'] > ndvi_comp['upper_bound']), np.nan, ndvi_comp['ndvi_c'])

    ########################
    #DLM
    ########################

    sm_sweep, sC_sweep, snu_sweep, FF_sweep, *_ = run_dlm(x_sweep.values, anCLM, vid, 1, rseas, fs1_deltas)
    sm_modcomp, sC_modcomp, snu_modcomp, FF_modcomp, *_ = run_dlm(ndvi_comp["ndvi_c_filt"].values.ravel(), anCLM_c, vid, 16, rseas, fs16_deltas)

    sweep_lbounds, sweep_ubounds = calculate_bounds(sm_sweep, sC_sweep, snu_sweep, quantile)
    comp_lbounds, comp_ubounds = calculate_bounds(sm_modcomp, sC_modcomp, snu_modcomp, quantile)

    gcc_z = (cosmos_df["GCC"] - np.nanmean(cosmos_df["GCC"])) / np.std(cosmos_df["GCC"])
    gcc_scale = gcc_z * np.nanstd(x_sweep) + np.nanmean(x_sweep)

    ndvi_comp_v = ndvi_comp['ndvi_c'].values.ravel()
    
    cosmos_df['roll_rain'] = cosmos_df['precip'].rolling(window=30, min_periods=1).sum()

    def ews(quant, sm, sC, snu, warmup, fs, n):
        # Calculate bd2 for the entire series
        bd2 = list(map(lambda m, C, nu: m + np.sqrt(C) * tdstr.ppf(quant, nu), sm, sC, snu))
        
        ews_indices = []

        n = int(n / fs)

        # Iterate over the series starting from 'warmup'
        for i in range(warmup, len(sm)):
            # Make sure the slice starts from a valid index (at least 0)
            start_idx = max(i - n, 0)
            
            # Look at the previous 'n' steps (or as many as available if near the start)
            historical_window = bd2[start_idx:i]

            # Calculate the median over the 'n' previous steps
            median_bd2 = np.median(historical_window)

            # Append index if sm[i] is greater than the median of bd2 over the last 'n' steps
            if sm[i] > median_bd2:
                ews_indices.append(i)

        return np.array(ews_indices)


    sweep_ews_ind = ews(quantile, sm_sweep, sC_sweep, snu_sweep, 365*2, 1, 365)
    comp_ews_ind = ews(quantile, sm_modcomp, sC_modcomp, snu_modcomp, 22*2, 16, 365)

    sweep_ews = np.full(len(x_sweep), np.nan)
    if len(sweep_ews_ind != 0):
        sweep_ews[sweep_ews_ind] = sm_sweep[sweep_ews_ind]

    comp_ews = np.full(len(ndvi_comp_v), np.nan)
    if len(comp_ews_ind != 0):
        comp_ews[comp_ews_ind] = sm_modcomp[comp_ews_ind]

    ########################
    #PLOT SIGNAL LOCATION
    ########################
    dates = cosmos_df["dt"]
    year_starts = dates[dates.dt.dayofyear == 1].index
    year_labels = dates.dt.year[dates.dt.dayofyear == 1]
    ytick_positions = np.linspace(-1, 20, num=11)  # -1 to 20 with 6 points
    ytick_labels = np.linspace(0, 1, num=11) 
    #ytick_labels = np.insert(ytick_labels, 0, 0)
    xlim = [dates.min(), dates.max()]

    import matplotlib.colors as mcolors
    grey_cmap = mcolors.LinearSegmentedColormap.from_list("", ["#F0F0F0", "darkgrey"])

    point_size = 0.7

    #plot
    fig, ax = plt.subplots(2, 1, figsize=(6, 4), gridspec_kw={'height_ratios': [1, 1.5]}) #in inches
    fig.subplots_adjust(wspace=0.08, hspace=0.08) 

    #plot ndvi series
    ax[0].scatter(dates, cosmos_df["ndvi_d"], c="lightgrey",  label="MOD09GQ", s = point_size)
    ax[0].scatter(dates, x_sweep_sig, c="black",  label="Sweep (sig.)", s = point_size) 
    ax[0].scatter(dates, x_sweep_nonsig, color='black',  label="Sweep (non-sig.)", marker='+', s=point_size+5,  linewidths=0.3)
    ax[0].plot(dates, loc_mean_if, color = "black", label = "Phenology center", linewidth = 0.9)
    ax[0].plot(dates, loc_max_if, color = "black",  label = "Phenology envelope", linewidth = 0.6, linestyle='dashed')
    ax[0].plot(dates, loc_min_if, color = "black", linewidth = 0.6, linestyle='dashed')
    ax[0].legend(loc='lower right', fontsize = 6, ncol=2)
    ax[0].set_xlim(xlim)

    #plot annual signal location (defines mean)
    interval = 0.015
    round_lvl = 0.05
    l_min = math.floor(np.nanmin(center_mat_w) / round_lvl) * round_lvl
    l_max = math.floor(np.nanmax(center_mat_w) / round_lvl) * round_lvl
    levels = np.arange(l_min, l_max, interval)
    contour = ax[1].contour(center_mat_w, origin='lower', levels = levels, cmap = grey_cmap, linewidths = 0.8)
    #ax[1].contour(center_mat, origin='lower', levels = [sig_wp], linewidths = 1, linestyles='dashed', colors = "black")
    ax[1].plot(np.arange(len(best_center_ind_if)), best_center_ind_if + 0.5, color = "black", linewidth = 0.9, label = "Selected phenology center")
    #ax[1].plot(np.arange(len(best_center_ind_sig)), best_center_ind_sig + 0.5, color = "black", linewidth =  0.8, label = "Seasonal center (*)")
    #custom_line = Line2D([0], [0], color='black', lw=1, linestyle='--', label='95% CI power')
    
    # Get existing legend handles and labels
    handles, labels = ax[1].get_legend_handles_labels()

    # Append the custom legend entry
    #handles.append(custom_line)
    #labels.append('95% CI Wavelet Power')

    # Add the combined legend to the bottom subplot
    ax[1].legend(handles=handles, labels=labels, loc='lower right', fontsize=6, ncol=3)

    norm = mcolors.Normalize(vmin = l_min, vmax = l_max)
    sm = cm.ScalarMappable(cmap=grey_cmap, norm=norm)
    sm.set_array([])
    
    pos = ax[1].get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.01, pos.height])  # Adjust the 0.01 and 0.02 values to position and size the color bar
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.ax.set_ylabel('Wavelet Power Contours', rotation=270, labelpad=-5, fontsize=6, verticalalignment='center')

    # Customize color bar ticks and labels
    cbar.set_ticks([l_min, l_max])
    cbar.set_ticklabels([l_min, l_max], fontsize = 5)
 
    ax[1].set_xlim(0, len(x)-1)
    ax[1].set_xticks(year_starts)
    ax[1].set_xticklabels(year_labels, rotation=0, ha='center', fontsize = 6)
    ax[1].set_yticks(ytick_positions)
    ax[1].set_yticklabels(np.round(ytick_labels, 1), rotation=0, ha='right')
    valid_rows = ~np.isnan(center_mat).any(axis=1)
    min_index = np.where(valid_rows)[0][0]  # First index with no NaNs
    max_index = np.where(valid_rows)[0][-1]  # Last index with no NaN

    ax[1].set_ylim(min_index+1, max_index)
    #ax[1].grid(True)  # Add gridlines (optional)
    #ax[1].legend(loc='lower right', fontsize = 6,  ncol=3)

    xlim = [dates.min(), dates.max()]

    ax[0].set_ylabel('NDVI', fontsize=6)
    ax[1].set_ylabel('Potential centers', fontsize=6)

    ax[0].tick_params(axis='both', which='major', labelsize=5)
    ax[1].tick_params(axis='both', which='major', labelsize=5)


    for ax_i in ax[:-1]:
        plt.setp(ax_i.get_xticklabels(), visible=False)
        ax_i.tick_params(axis='x', which='both', bottom=False, top=False)

    path_name = f"data_out/figs/sig_location/{site_data['site_name']}_sigloc.png"
    output_path = path_name  # specify your file path and name here
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)

    ########################
    #PLOT SIGNAL DENOISING
    ########################

    x_subset = x.copy()
    na_ind = (x_subset > loc_max_if) | (x_subset < loc_min_if)
    x_subset[na_ind] = np.nan
    levels = np.linspace(0, np.max(np.abs(wx_og))*0.7, 10)
    levels = np.append(levels, np.max(np.max(np.abs(wx_og))))

    wx, scales, *_ = ssq.cwt(x, wav, fs=fs)
    freqs = scale_to_freq(scales, wav, len(x), fs=fs)
    periods = 1/freqs

    og_av = np.mean(np.abs(wx_og), axis = 1)
    filt_av = np.mean(np.abs(wx_filt), axis = 1)

    labels = (2, 4, 8, 16, 32, 64, 128, 365)

    indices = []
    for label in labels:
        index = np.argmin(np.abs(label - periods))
        indices.append(index)

    grey_cmap = mcolors.LinearSegmentedColormap.from_list("grey_cmap", ["#F0F0F0", "black"])

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes



    fig, ax = plt.subplots(2, 3, figsize=(6, 3), gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [1, 0.4, 0.15]})

    # Adjust the spacing between columns using fig.subplots_adjust
    fig.subplots_adjust(wspace=0.05, hspace = 0.05)  # Default spacing between col 1 and 2

    # Manually adjust the spacing between columns 2 and 3
    for ax_i in [ax[0, 2], ax[1, 2]]:
        pos = ax_i.get_position()
        pos.x0 -= 0.00  # Shift left side closer to the center column
        pos.x1 -= 0.02  # Shift right side closer to the center column
        ax_i.set_position(pos)

    levels = np.linspace(np.min(np.abs(wx_og)), np.max(np.abs(wx_og)), 10)

    #plot ndvi series
    ax[0, 0].scatter(dates, x_subset, c="black",  label="Phenology envelope", s = point_size)
    #ax[0, 0].plot(dates, loc_max_if, color = "black", linewidth = 0.6)
    #ax[0, 0].plot(dates, loc_min_if, color = "black",  linewidth = 0.6)
    ax[0, 0].set_ylabel('Enveloped NDVI', fontsize = 6)

    ax[1, 0].scatter(dates, x_sweep, c="black",  label="Sweep phenology", s = point_size)
    ax[1, 0].set_xlabel('Year', fontsize = 6)
    ax[1, 0].set_ylabel('Noise-supressed NDVI', fontsize = 6)

    ax[0, 1].contourf(np.abs(wx_og), aspect = 'auto', levels = levels, cmap = grey_cmap)
    ax[0, 1].axhline(y=np.argmin(np.abs(l_period - periods)), color='black', linestyle='--', linewidth=0.6)
    ax[0, 1].set_yticks(indices, labels)

    
    contour = ax[1, 1].contourf(np.abs(wx_filt), aspect = 'auto', levels = levels, cmap = grey_cmap)
    ax[1, 1].axhline(y=np.argmin(np.abs(l_period - periods)), color='black', linestyle='--', linewidth=0.6)
    ax[1, 1].set_yticks(indices, labels)
    ax[1, 1].set_xticks(year_labels.iloc[[0, -1]].index)
    ax[1, 1].set_xticklabels(year_labels.iloc[[0, -1]], rotation=0, ha='center', fontsize = 6)
    ax[1, 1].set_xlabel('Year', fontsize = 6)
    # Create an inset axis for the colorbar

    norm = mcolors.Normalize(vmin = np.min(levels), vmax = np.max(levels))
    sm = cm.ScalarMappable(cmap=grey_cmap, norm=norm)
    sm.set_array([])
    
    pos = ax[1, 1].get_position()
    cbar_ax = fig.add_axes([pos.x1 - 0.185, pos.y0 + 0.02, 0.006, 0.1])  # Adjust the 0.01 and 0.02 values to position and size the color bar
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    #cbar.ax.set_ylabel('Wavelet Power', labelpad=-5, fontsize=6, verticalalignment='center')
    cbar.set_ticks([np.min(levels), np.max(levels)])
    cbar.set_ticklabels([np.round(np.min(levels), 3), np.round(np.max(levels), 2)], fontsize = 5)
    #cbar.set_label('Wavelet\nPower', rotation=0, labelpad=15, fontsize = 6, ha = "left")
    cbar.ax.text(-0.1, 1.45, 'Wavelet\nPower', rotation=0, va='center', ha='left', fontsize = 6, transform=cbar.ax.transAxes)
    cbar.ax.xaxis.label.set_position((0, 1))
    cbar.ax.xaxis.label.set_ha('left') 

    factor_v = factor_mask[:, 0]

    ax[0, 2].plot(og_av, np.arange(len(og_av)), color = "black", linewidth=0.6)
    #ax[0, 2].plot(factor_v, np.arange(len(factor_v)), linestyle='--', linewidth=0.6, color = "black")
    ax[0, 2].axhline(y=np.argmin(np.abs(l_period - periods)), color='black', linestyle='--', linewidth=0.6)
    ax[0, 2].set_yticks(indices, labels)
    ax[0, 2].set_ylim(0, wx_filt.shape[0])
    ax[0, 2].set_ylabel('Period (years)', fontsize = 6)
    ax[0, 2].yaxis.set_label_position("right")
    ax[0, 2].yaxis.tick_right()

    ax[1, 2].plot(filt_av, np.arange(len(filt_av)), label = "Average", color = "black", linewidth=0.6)
    #ax[1, 2].plot(factor_v, np.arange(len(factor_v)), linestyle='--', linewidth=0.6, color = "black", label = "Factor")
    ax[1, 2].axhline(y=np.argmin(np.abs(l_period - periods)), color='black', linestyle='--', linewidth=0.6)
    ax[1, 2].set_yticks(indices, labels)
    ax[1, 2].set_ylim(0, wx_filt.shape[0])
    ax[1, 2].set_xlabel('Av. Power', fontsize = 6)
    ticks = ax[1, 2].get_xticks()
    ax[1, 2].set_xticklabels([f'{tick:.1f}' for tick in ticks])
    #ax[1, 2].legend(fontsize = 6)
    ax[1, 2].set_ylabel('Period (years)', fontsize = 6)
    ax[1, 2].yaxis.set_label_position("right")
    ax[1, 2].yaxis.tick_right()

    for ax_i in ax.flat:
        ax_i.tick_params(axis='both', which='major', labelsize=6)

    for ax_i in [ax[0, 1], ax[1, 1]]:
        plt.setp(ax_i.get_yticklabels(), visible=False)
        ax_i.tick_params(axis='y', which='both', left=False, top=False)

    for ax_i in [ax[0, 0], ax[0, 1], ax[0, 2]]:
        plt.setp(ax_i.get_xticklabels(), visible=False)
        ax_i.tick_params(axis='x', which='both', bottom=False, top=False)

    path_name = f"data_out/figs/denoising/{site_data['site_name']}_denoise.png"
    plt.savefig(path_name, dpi=300, bbox_inches='tight', pad_inches=0)




    ########################
    #PLOT DLM RESULTS
    ########################

    fig, ax = plt.subplots(3, 1, figsize=(4, 4), gridspec_kw={'width_ratios': [1], 'height_ratios': [1, 1, 2]})

    # First subplot
    ax[0].scatter(cosmos_df["dt"][sig_vector], x_sweep[sig_vector], color='black', label='SWEEP (sig.)', s=point_size)
    ax[0].scatter(cosmos_df["dt"][~sig_vector], x_sweep[~sig_vector], color='black', label='SWEEP (non-sig.)', marker='+', s=point_size+5,  linewidths=0.3)
    ax[0].scatter(ndvi_comp["dt"], ndvi_comp['ndvi_c_filt'], color='red', label='Comp.', s=point_size)  # Changed label for clarity
    ax[0].legend(loc='lower right', fontsize = 6, ncol = 3)
    
    ax[0].set_ylabel('NDVI', fontsize = 6)
    ax[0].tick_params(axis='x', labelsize=6) 
    ax[0].tick_params(axis='y', labelsize=6) 

    # Second subplot
    lgn1 = ax[1].plot(cosmos_df["dt"], cosmos_df['cosmos_vwc'], color='black', label='Volumetric water content (%)', linewidth = 0.6, zorder=1)
    ax[1].set_ylabel('Soil moisture (%)', fontsize = 6)
    ax[1].tick_params(axis='x', labelsize=6) 
    ax[1].tick_params(axis='y', labelsize=6) 

    #third subplot

    comp_lbounds_speed = -comp_lbounds
    comp_ubounds_speed = -comp_ubounds
    speed_modcomp = -sm_modcomp
    speed_comp_ews =-comp_ews

    ax[2].fill_between(ndvi_comp["dt"], comp_lbounds_speed, comp_ubounds_speed, facecolor='red', alpha = 0.2, zorder=1)
    ax[2].plot(ndvi_comp["dt"], speed_modcomp, color = 'red', linestyle = 'dashed', label = 'Comp. Speed', linewidth= 0.6, zorder=2) 
    ax[2].plot(ndvi_comp["dt"], speed_comp_ews, color = 'red', label = 'Comp. CSD', linewidth= 2, zorder=3) 

    sweep_lbounds_speed =  -sweep_lbounds
    sweep_ubounds_speed =  -sweep_ubounds
    speed_sweep =  -sm_sweep
    speed_sweep_ews =  -sweep_ews

    ax[2].fill_between(cosmos_df["dt"], sweep_lbounds_speed, sweep_ubounds_speed, facecolor='lightgrey', alpha = 0.8, zorder=4)
    ax[2].plot(cosmos_df["dt"], speed_sweep, color = 'black', linestyle = 'dashed', label = 'SWEEP Speed', linewidth= 0.6, zorder=5) 
    ax[2].plot(cosmos_df["dt"], speed_sweep_ews, color = 'black',  label = 'SWEEP CSD', linewidth= 2, zorder=6)  

    ax[2].set_ylim(-1, 1)
    ax[2].legend(loc='upper right', fontsize = 6, ncol = 2)
    ax[2].set_ylabel('System Speed (-AC1)', fontsize = 6)
    ax[2].set_xlabel('Date', fontsize = 6)
    ax[2].tick_params(axis='x', labelsize=6) 
    ax[2].tick_params(axis='y', labelsize=6) 

    for ax_i in [ax[0], ax[1]]:
        plt.setp(ax_i.get_xticklabels(), visible=False)
        ax_i.tick_params(axis='x', which='both', bottom=False, top=False)
                       
    path_name = f"data_out/figs/dlm/{site_data['site_name']}_dlm.png"
    plt.savefig(path_name, dpi=300, bbox_inches='tight', pad_inches=0)

    ########################
    #PLOT PHENOLOGY COMPARE
    ########################
    fig, ax = plt.subplots(1, 1, figsize=(6, 2))

    # Plotting on the first axis
    ax.scatter(cosmos_df["dt"][sig_vector], x_sweep[sig_vector], color='black', label='Sweep (sig.)', s=point_size)
    ax.scatter(cosmos_df["dt"][~sig_vector], x_sweep[~sig_vector], color='black', label='Sweep (non-sig.)', marker='+', s=point_size+5,  linewidths=0.3)
    ax.scatter(ndvi_comp["dt"], ndvi_comp['ndvi_c_filt'], color='red', label='Mod13Q1', s=2)  # Changed label for clarity
    ax.legend(loc='lower right', fontsize=6, ncol=2)

    # Setting y-axis for ax
    ax.set_ylim(-0.5, 1.2)
    yticks1 = np.arange(0.2, 1.1, 0.1)
    ax.set_yticks(yticks1)
    ax.set_yticklabels([f'{tick:.2f}' for tick in yticks1])

    # Creating a secondary y-axis
    ax2 = ax.twinx()
    ax2.scatter(cosmos_df["dt"], cosmos_df["GCC"], color="green", s=1, label = 'GCC')

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    ax.legend(lines + lines2, labels + labels2, loc='upper left', fontsize = 6)

    # Setting y-axis for ax2
    ax2.set_ylim(0.3, 0.5)
    yticks2 = np.arange(0.3, 0.42, 0.02)
    ax2.set_yticks(yticks2)
    ax2.set_yticklabels([f'{tick:.2f}' for tick in yticks2])

    ax.set_ylabel('NDVI', fontsize = 6)
    ax2.set_ylabel('GCC', fontsize = 6)

    ax.tick_params(axis='x', labelsize=6) 
    ax.tick_params(axis='y', labelsize=6) 
    ax2.tick_params(axis='y', labelsize=6) 

    ax.yaxis.set_label_coords(-0.08, 0.65)
    ax2.yaxis.set_label_coords(1.08, 0.25)

    path_name = f"data_out/figs/compare/{site_data['site_name']}_compare.png"
    plt.savefig(path_name, dpi=300, bbox_inches='tight', pad_inches=0)

    ########################
    #PLOT ATTRACTOR SURFACES
    ########################

    dates = cosmos_df['dt']
    plot_params = surface_plot_params.iloc[i]
    site_name = site_data['site_name']
    warmup = int(plot_params[3])

    file_name = f"attr_surface_{plot_params[0]}.png"
    plot_landscape(FF_sweep, warmup, dates, plot_params, file_name)

    ########################
    #PREPARE TABLES
    ########################

    def min_max_normalize(series):
        return (series - np.min(series)) / (np.max(series) - np.min(series))

    #calc gcc season info
    if cosmos_df['GCC'].notna().any():

        sweep_gcc_wave_xy, sweep_gcc_coh, sweep_gcc_cpr, sweep_gcc_phase, sweep_gcc_phaseday = wave_variance(cosmos_df['GCC'], x_sweep, 1, u_period, l_period)
        comp_gcc_wave_xy, comp_gcc_coh, comp_gcc_cpr, comp_gcc_phase, comp_gcc_phaseday = wave_variance(ndvi_comp['GCC'], ndvi_comp['ndvi_c_filt'],  16, u_period, l_period)

        mask = ~np.isnan(ndvi_comp['GCC']) & ~np.isnan(ndvi_comp['ndvi_c_filt'])
        filtered_series1 = min_max_normalize(ndvi_comp['GCC'][mask])
        filtered_series2 = min_max_normalize(ndvi_comp['ndvi_c_filt'][mask])
        y_mean = np.mean(filtered_series1)
        TSS = np.sum((filtered_series1 - y_mean) ** 2)
        RSS = np.sum((filtered_series1 - filtered_series2) ** 2)
        comp_r2 = 1 - (RSS / TSS)

        mask = ~np.isnan(cosmos_df['GCC']) & ~np.isnan(x_sweep)
        filtered_series1 = min_max_normalize(cosmos_df['GCC'][mask])
        filtered_series2 = min_max_normalize(x_sweep[mask])
        y_mean = np.mean(filtered_series1)
        TSS = np.sum((filtered_series1 - y_mean) ** 2)
        RSS = np.sum((filtered_series1 - filtered_series2) ** 2)
        sweep_r2 = 1 - (RSS / TSS)                    
    else: 
        comp_r2= np.nan
        comp_gcc_coh = np.nan
        comp_gcc_cpr = np.nan
        comp_gcc_phaseday = np.nan
        sweep_r2 = np.nan
        sweep_gcc_coh = np.nan
        sweep_gcc_cpr = np.nan
        sweep_gcc_phaseday = np.nan
    
    #calc start of EWS
    sweep_ews_boo = ~np.isnan(sweep_ews)
    transition_to_true = np.where((sweep_ews_boo[:-1] == False) & (sweep_ews_boo[1:] == True))[0] + 1
    dates_to_true = cosmos_df['dt'][transition_to_true]
    if len(dates_to_true) == 0:
        comp_ews_starts = 'None'
    else:
        sweep_ews_starts = [date.strftime("%d/%m/%Y") for date in dates_to_true][0]
    
    comp_ews_boo = ~np.isnan(comp_ews)
    transition_to_true = np.where((comp_ews_boo[:-1] == False) & (comp_ews_boo[1:] == True))[0] + 1
    dates_to_true = cosmos_df['dt'][transition_to_true]
    if len(dates_to_true) == 0:
        comp_ews_starts = 'None'
    else:
        comp_ews_starts = [date.strftime("%d/%m/%Y") for date in dates_to_true][0]

    #DLM FIT
    from scipy.stats import pearsonr

    sweep_a = x_sweep - np.nanmean(x_sweep)
    sweep_pred = FF_sweep['predictions']
    sweep_a = sweep_a[warmup: ]
    sweep_pred = sweep_pred[warmup: ]

    mask = ~np.isnan(sweep_a[1:])
    sweep_rmse = np.round(np.sqrt(np.mean(np.square(sweep_a[1:][mask] - sweep_pred[mask]))), 3)

    Y = np.array(sweep_a[1:][mask])
    predictions = np.array(sweep_pred[mask])
    #mean_observed = np.nanmean(Y)
    #ss_tot = np.sum((Y - mean_observed) ** 2)
    #ss_res = np.sum((Y - predictions) ** 2)
    #sweep_r2_model = np.round(1 - (ss_res / ss_tot), 3)

    sweep_r2_model = np.round(pearsonr(Y[10:], predictions[10:])[0] ** 2, 3)
    sweep_lik = pd.Series(FF_sweep['slik'][warmup:])
    sweep_lik = sweep_lik.mean()
    
    comp_warmup = int(warmup / 16)

    comp_a = ndvi_comp['ndvi_c_filt'] - np.nanmean(ndvi_comp['ndvi_c_filt'])
    comp_pred = FF_modcomp['predictions']
    comp_a = comp_a[comp_warmup: ]
    comp_pred = comp_pred[comp_warmup: ]
    mask = ~np.isnan(comp_a[1:])
    comp_rmse = np.round(np.sqrt(np.mean(np.square(comp_a[1:][mask] - comp_pred[mask]))), 3)

    Y = np.array(comp_a[1:][mask])
    predictions = np.array(comp_pred[mask])
    #mean_observed = np.nanmean(Y)
    #ss_tot = np.sum((Y - mean_observed) ** 2)
    #ss_res = np.sum((Y - predictions) ** 2)
    #comp_r2_model = np.round(1 - (ss_res / ss_tot), 3)
    comp_r2_model = np.round(pearsonr(Y, predictions)[0] ** 2, 3)
    comp_lik = pd.Series(FF_modcomp['slik'][comp_warmup: ])
    comp_lik = comp_lik.mean()


    ndvi_comp['year'] = ndvi_comp['dt'].dt.year
    cosmos_df['year'] = cosmos_df['dt'].dt.year
    cosmos_df['sweep'] = x_sweep

    name = [site_data['site_name']]

    df_non_nan = cosmos_df.dropna(subset=['x_sweep'])
    intervals = df_non_nan['dt'].diff().dropna()
    sweep_interval = round(intervals.mean() / pd.Timedelta(days=1), 2)

    df_non_nan = ndvi_comp.dropna(subset=['ndvi_c_filt'])
    intervals = df_non_nan['dt'].diff().dropna()
    comp_interval = round(intervals.mean() / pd.Timedelta(days=1), 2)

    site_sweep = pd.DataFrame({
        'Site': name,
        'comp int': comp_interval,                          
        'comp cpr': np.round(comp_gcc_cpr, 2),
        'sweep int': sweep_interval,
        'sweep cpr': np.round(sweep_gcc_cpr, 2)})

    site_dlm = pd.DataFrame({'Site': name, 
                 'comp RMSE': comp_rmse,
                 'comp R2': comp_r2_model,
                 'comp ar1 var': np.var(sm_modcomp),
                 'sweep RMSE': sweep_rmse,
                 'sweep R2': sweep_r2_model,
                 'sweep ar1 var': np.var(sm_sweep),           
                 })
    
    sweep_results_table = pd.concat([sweep_results_table, site_sweep], ignore_index=True)
    dlm_results_table = pd.concat([dlm_results_table, site_dlm], ignore_index=True)



sweep_results_table.to_csv('sweep_results.csv', index=False)
dlm_results_table.to_csv('dlm_results.csv', index=False)

sweep_latex = sweep_results_table.to_latex(index=False, float_format="%.2f")
dlm_latex = dlm_results_table.to_latex(index=False, float_format="%.2f")

print(sweep_latex)
print(dlm_latex)

#PLOT GIFS

#glenwherry
i = 18
warmup = int(365 * 1.5)

#stiperstones
i = 44
warmup = int(365 * 2)

#moor house
i = 30
warmup = int(365 * 1.5)

#gisbun forest
i = 16
warmup = int(365 * 2.4)




site_data = cosmos_sites_df.iloc[i]
plot_title = f"Site: {site_data['site_name']}, Land use: {site_data['land_cover']}"
code = cosmos_sites_df.index[i]

cosmos_df = cosmos_data[code]

x = cosmos_df["ndvi_d"]
x = x.values.ravel()

signal_index, signal_location, best_power, sig_vector, av_power_mat, power_mat, center_mat, range_mat, center_mat_w, range_mat_w = sweep_locate(x, win_mean, win_range, wav, fs, l_period, u_period, min_thresh, sig_wp, cost_i, buff)
signal_index_infil, signal_index_sigonly, signal_location_infil, signal_location_sigonly = sweep_infil(signal_index, signal_location, best_power, sig_vector, av_power_mat, win_mean, win_range, buff)

#SWEEP EXTRATIONS
x_sweep, wx_og, wx_filt, factor_mask = sweep_extract(x, signal_location_infil, wav, fs, l_period, exp)
cosmos_df['x_sweep'] = x_sweep

########################
#DLM - PREPROCESSING
########################

#identify missing dates in GCC
missing_dates = cosmos_df[cosmos_df['GCC'].isna()]['dt'].tolist()
sweep_rm = cosmos_df["dt"].isin(missing_dates)
comp_rm = ndvi_comp["dt"].isin(missing_dates)

#calculate climate anomalies for daily
cosmos_df['dt'] = pd.to_datetime(cosmos_df['dt'])
cosmos_df['month'] = cosmos_df['dt'].dt.month
monthly_mean_precip = cosmos_df.groupby(['month'])['precip'].transform('mean')
cosmos_df['precip_an'] = cosmos_df['precip'] - monthly_mean_precip
cosmos_df = cosmos_df.fillna({'precip_an':0})
anCLM = cosmos_df['precip_an']

########################
#DLM
########################
rseas = [1, 2] 
vid = 2 # index of autocorrelation
fs1_deltas = np.ones(4) * 0.996

sm_sweep, sC_sweep, snu_sweep, FF_sweep, *_ = run_dlm(x_sweep.values, anCLM, vid, 1, rseas, fs1_deltas)

dates = cosmos_df['dt']
plot_params = surface_plot_params.iloc[i]
site_name = site_data['site_name']
warmup = int(plot_params[3])

file_name = f"attr_surface_{plot_params[0]}.png"
plot_landscape(FF_sweep, warmup, dates, plot_params, file_name)


#code to produce GIF

########################
#PLOT ATTRACTOR SURFACES GIF
########################




plot_landscape_gif(FF_sweep, warmup, dates, plot_params)



#Time series plot
import matplotlib.pyplot as plt
import os
import imageio


n = 5
warmup_n = int(warmup / n)

quantile = 0.9
sweep_lbounds, sweep_ubounds = calculate_bounds(sm_sweep, sC_sweep, snu_sweep, quantile)
sweep_ews_ind = ews(quantile, sm_sweep, sC_sweep, snu_sweep, 365*2, 1, 365)

sweep_ews = np.full(len(x_sweep), np.nan)
if len(sweep_ews_ind != 0):
    sweep_ews[sweep_ews_ind] = sm_sweep[sweep_ews_ind]


# Define directories
time_series_frames_dir = "C:/data/gif_frames"
os.makedirs(time_series_frames_dir, exist_ok=True)

point_size = 3

# Render time series frames sequentially
def render_time_series_frames():
    for i in range(warmup_n, int(len(dates)/n)):

        current_date = dates.iloc[i * n]
        fig, ax = plt.subplots(3, 1, figsize=(5, 5), gridspec_kw={'width_ratios': [1], 'height_ratios': [1, 1, 1]})
        
        ax[0].scatter(cosmos_df["dt"], cosmos_df['ndvi_d'],  c="lightgrey",  label="Daily NDVI", s = point_size)   
        ax[0].scatter(cosmos_df["dt"], x_sweep, color='black', label='SWEEP NDVI', s=point_size)        
        #ax[0].scatter(ndvi_comp["dt"], ndvi_comp['ndvi_c_filt'], color='red', label='Mod13Q1', s=point_size)  # Changed label for clarity
        ax[0].axvline(x=current_date, color="red", linestyle="--", label="Current Time")
        ax[0].legend(loc='lower right', fontsize = 6, ncol = 3)
        #ax[0].grid(True, linewidth = 0.5) 

        ax[0].set_ylabel('NDVI', fontsize = 6)
        ax[0].tick_params(axis='x', labelsize=6) 
        ax[0].tick_params(axis='y', labelsize=6) 

        # Second subplot
        ax[1].plot(cosmos_df["dt"], cosmos_df['cosmos_vwc'], color='black', linewidth = 0.3, zorder=1)
        ax[1].set_ylabel('Soil moisture (%)', fontsize = 6)
        ax[1].tick_params(axis='x', labelsize=6) 
        ax[1].tick_params(axis='y', labelsize=6) 
        ax[1].axvline(x=current_date, color="red", linestyle="--", label="Current Time")

        sweep_lbounds_speed =  -sweep_lbounds
        sweep_ubounds_speed =  -sweep_ubounds
        speed_sweep =  -sm_sweep
        speed_sweep_ews =  -sweep_ews

        #third subplot
        ax[2].fill_between(cosmos_df["dt"], sweep_lbounds_speed, sweep_ubounds_speed, facecolor='lightgrey', alpha = 0.8, zorder=4)
        ax[2].plot(cosmos_df["dt"], speed_sweep, color = 'black', linestyle = 'dashed', label = 'System speed', linewidth= 0.6, zorder=5) 
        ax[2].plot(cosmos_df["dt"], speed_sweep_ews, color = 'black',  label = 'Critical slowing down', linewidth= 2, zorder=6)  
        ax[2].axvline(x=current_date, color="red", linestyle="--", label="Current Time")

        ax[2].set_ylim(-0.7, 0.7)
        ax[2].legend(loc='upper right', fontsize = 6, ncol = 2)
        ax[2].set_ylabel('System speed (-AC)', fontsize = 6)
        ax[2].set_xlabel('Date', fontsize = 6)
        ax[2].tick_params(axis='x', labelsize=6) 
        ax[2].tick_params(axis='y', labelsize=6) 
        #ax[2].grid(True, linewidth = 0.5) 

        for ax_i in [ax[0], ax[1]]:
            plt.setp(ax_i.get_xticklabels(), visible=False)
            ax_i.tick_params(axis='x', which='both', bottom=False, top=False)

        # Save the frame
        frame_path = os.path.join(time_series_frames_dir, f"frame_{i:04d}.png")
        fig.savefig(frame_path, dpi=300)
        plt.close()

# Generate frames
render_time_series_frames()

# Combine frames into a GIF
output_gif = f"data_out/figs/gifs/time_series_animation_{site_data['site_name']}.gif"
with imageio.get_writer(output_gif, mode="I", fps=12, loop=0) as writer:
    for i in range(warmup_n, int(len(dates)/n)):
        frame_path = os.path.join(time_series_frames_dir, f"frame_{i:04d}.png")
        writer.append_data(imageio.imread(frame_path))

# Cleanup temporary frames
for frame_file in os.listdir(time_series_frames_dir):
    os.remove(os.path.join(time_series_frames_dir, frame_file))
os.rmdir(time_series_frames_dir)

print(f"Time series animation saved as '{output_gif}'.")


