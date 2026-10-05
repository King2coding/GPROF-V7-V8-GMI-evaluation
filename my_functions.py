#%% import required packages
import numpy as np
from glob import glob
import os 
import pandas as pd
from pyhdf.SD import SD, SDC
from pyhdf import HDF, VS, V
import h5py
from PIL import Image
from netCDF4 import Dataset
import datetime
import rasterio
from rasterio.transform import Affine
import xarray as xr
from scipy.interpolate import griddata
# import rioxarray as rxr
from scipy.signal import convolve2d



#%% functions
def dir_files(directory, extension):
    
    
    """"
    Returns an array of files in the specified directory that have the defined
    extension.
    
    :param directory: Directory of interest with directory separator at the end
    :param extension: File extension of interest
    :return: A one dimenstional array of files
    """

    files = np.array(glob(directory + '*.{}'. format(extension)))
    files.sort()
    return files

def AutoSnow_datetime(AutoSnow_file):
    
    """"
    Returns datetime index of the introduced AutoSnow file.
    
    :param AutoSnow_file: AutoSnow file
    :return: Datetime index of the AutoSnow file
    """
    
    filename = os.path.basename(AutoSnow_file)
    dt = pd.to_datetime(filename[21:29], format='%Y%m%d')
    return dt

def AutoSnow_HDF_SDreader(AutoSnow_file, variable):
    
    """"
    Returns variable data from the introduced AutoSnow file.
    
    :param AutoSnow_file: AutoSnow file
    :return: Data for variable of interest
    """
    
    sd = SD(AutoSnow_file, SDC.READ)
    sds_obj = sd.select(variable)
    data = sds_obj.get()
    sd.end()
    return data

def CloudSat_datetime(CloudSat_file):
    
    """"
    Returns datetime index of the introduced CloudSat file.
    
    :param CloudSat_file: CloudSat file
    :return: Datetime index of the CloudSat file
    """
    
    filename = os.path.basename(CloudSat_file)
    dt = pd.to_datetime(filename[:13], format='%Y%j%H%M%S')
    return dt


def CloudSat_datetime_(CloudSat_file):
    
    """"
    Returns datetime index of the introduced CloudSat file.
    
    :param CloudSat_file: CloudSat file
    :return: Datetime index of the CloudSat file
    """
    
    filename = os.path.basename(CloudSat_file)
    dt = datetime.datetime.strptime(filename[:13], '%Y%j%H%M%S')
    return dt

def CloudSat_HDF_SDreader(CloudSat_file, variable):
    
    """"
    Returns variable data from the introduced CloudSat file.
    
    :param CloudSat_file: CloudSat file
    :return: Data for variable of interest
    """
    
    sd = SD(CloudSat_file, SDC.READ)
    sds_obj = sd.select(variable)
    data = sds_obj.get()
    fill_value = sds_obj.attributes()['_FillValue']
    # data = data.astype(float) # This line might be probably needed. I add it.
    data[data == fill_value] = np.nan
    sd.end()
    return data

def CloudSat_HDF_HDreader(CloudSat_file, variable):
    
    """"
    Returns variable data from the introduced CloudSat file.
    
    :param CloudSat_file: CloudSat file
    :return: Data for variable of interest
    """
    
    missing_values = {
        'DEM_elevation':9999,
        'Vertical_binsize':-9999,
        'snow_retrieval_status':0,
        'snowfall_rate_sfc':-999,
        'snowfall_rate_sfc_confidence':-1
        }
    hd = HDF.HDF(CloudSat_file)
    vs = hd.vstart()
    vid = vs.find(variable)
    varid = vs.attach(vid)
    varid.setfields(variable)
    nrecs, _, _, _, _ = varid.inquire()
    data = varid.read(nRec=nrecs)
    data = np.array(data).squeeze()
    if (variable == 'snowfall_rate_sfc_confidence') | (variable == 'snowfall_rate_sfc'): # I added this line.
        data = data.astype('float32') # I added this line.
    if variable in missing_values:
        fill_value = missing_values[variable]
        data[data == fill_value] = np.nan
    varid.detach()
    hd.close()
    return data

def H5_write(path, filename, data): 
    
    """"
    Write HDF5 file in the path with variables in data dictionary.
    
    :param path: Directory to which HDF5 file will be saved with directory 
                separator at the end
    :param filename: Name of HDF5 file
    :param data: A dictionary containing the variables and corresponding data
    """
    
    fullname = path + filename + '.h5'
    if os.path.isfile(fullname):
        os.remove(fullname)
    with h5py.File(fullname, 'w') as f:
        keys = data.keys()
        for key in keys:
            info = data[key]
            f.create_dataset(key, data=info) 

def H5_reader(file, variable):
    
    """"
    Read variable from HDF5 file. It is suitable for files written by 
    HDF5_write method.
    
    :param file: file with path included
    :param variable: Name of variable
    :return data: Data included in the variable of the HDF5 file
    """
    
    with h5py.File(file, 'r') as h5:
        data = h5[variable][:]
    return data

def IMERG_file_datetime(IMERG_file):
    
    """"
    Returns datetime index of the introduced IMERG file.
    
    :param IMERG_file: IMERG file
    :return: Datetime index of the IMERG file
    """
    
    filename = os.path.basename(IMERG_file)
    dt = pd.to_datetime(filename.split('.')[4][:8], format='%Y%m%d')
    return dt

def tif_to_array(tif_file):
    
    """"
    Returns tif file as a numpy array.
    
    :param tif_file: Tif file
    :return: Equivalent numpy array
    """
    
    image = Image.open(tif_file)
    array = np.array(image)
    return array

def IMERG_NCreader(IMERG_file, variable):
    
    """"
    Returns variable data from the introduced IMERG file.
    
    :param IMERG_file: IMERG file
    :return: Data for variable of interest
    """
    
    with Dataset(IMERG_file) as nc:
        data = nc[variable][:]
        data.data[data.mask]=np.nan
        fill_value = nc[variable].getncattr('_FillValue')
        data.data[data == fill_value] = np.nan
    return data.data.squeeze()

def MHS_datetime(MHS_file):
    
    """"
    Returns datetime index of the MHS file.
    
    :param MHS_file: MHS file
    :return: Datetime index of the MHS file
    """
    
    filename = os.path.basename(MHS_file)
    date = filename[25:33]
    start_time = filename[35:41]
    end_time = filename[43:49]
    dt_start = pd.to_datetime(date+start_time, format='%Y%m%d%H%M%S')
    dt_end = pd.to_datetime(date+end_time, format='%Y%m%d%H%M%S')
    if dt_end < dt_start:
        #files which ending time is few minutes after 00:00 pm
        dt_end = dt_end + pd.Timedelta(1, 'day')
    return dt_start, dt_end

def MERRA2_datetime(MERRA2_file):
    
    """"
    Returns datetime index of the MERRA2 file.
    
    :param MERRA2_file: MERRA2 file
    :return: Datetime index of the MERRA2 file
    """
    
    filename = os.path.basename(MERRA2_file)
    dt = pd.to_datetime(filename[27:35], format='%Y%m%d').date()
    return dt

def MERRA2_datetime_(MERRA2_file):
    
    """"
    Returns datetime index of the MERRA2 file.
    
    :param MERRA2_file: MERRA2 file
    :return: Datetime index of the MERRA2 file
    """
    
    dt = os.path.basename(MERRA2_file)[27:35]
    return dt

def MERRA2_HDF_SDreader(MERRA2_file, field):
    
    """"
    Returns filed data from the MERRA2 file.
    
    :param MERRA2_file: MERRA2 file
    :return: Data for the filed of interest
    """
    import os
    print("Opening:", MERRA2_file)
    print("Exists?:", os.path.exists(MERRA2_file))
    print("Is file?:", os.path.isfile(MERRA2_file))
    sd = SD(MERRA2_file, SDC.READ)
    dset = sd.select(field)
    data = dset.get()
    attrs = dset.attributes(full=1)
    if 'missing_value' in attrs:
        fval = attrs['missing_value'][0]
        data[data==fval] = np.nan
    sd.end()
    return data

def AVHRR_datetime(AVHRR_file):
    basename = os.path.basename(AVHRR_file)
    st_str = basename[13:18] + basename[20:24]
    et_str = basename[13:18] + basename[26:30]
    st = pd.to_datetime(st_str, format='%y%j%H%M')
    et = pd.to_datetime(et_str, format='%y%j%H%M')
    if et < st:
        #files which ending time is few minutes after 00:00 pm
        et = et + pd.Timedelta(1, 'day')
    return st, et

def AVHRR_datetime_NC_files(AVHRR_file):

    name_splited = os.path.basename(AVHRR_file).split('.')

    yr_DOY = name_splited[3][1:]
    st_time = name_splited[4][1:]
    end_time = name_splited[5][1:]
    yr = datetime.datetime.strptime(yr_DOY, '%y%j').year

    st_dt = datetime.datetime.strptime(yr_DOY + st_time, '%y%j%H%M')

    end_dt = datetime.datetime.strptime(yr_DOY + end_time, '%y%j%H%M')

    if end_dt < st_dt:
        #files which ending time is few moments after 00:00 pm
        end_dt = end_dt + datetime.timedelta(days=1)

    return st_dt, end_dt, yr

def gridder(ref_grid, grid_resolution, lon, lat, variable):

    """"
    Returns grided data from 1-D arrays of latitudes, longitudes, and measured variable
    
    :param resolution: the final resolution of the grided data
    :param lon, lat: the longitude and latitude of measured variable (1-D array)
    :param variable: the value of the measured variable (1-D array)
    :return: grided dataset over the whole world with desired resolution
    """

    idx = np.digitize(lon, np.arange(-180, 180.1, grid_resolution).round(1)) - 1
    idy = np.digitize(lat, np.arange(-90, 90.1, grid_resolution).round(1)) - 1
    df = pd.DataFrame({'x':idx, 'y':idy, 'z':variable})
    df = df[(df.x != -1) | (df.y != -1)].copy()
    mean_df = df.groupby(['x', 'y']).mean()
    ref_grid[mean_df.index.get_level_values(1), mean_df.index.get_level_values(0)] = mean_df['z']
    
    return ref_grid


def reduce_mem_usage(df):
    # start_mem = df.memory_usage().sum() / 1024**2
    # print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        if (col == 'scan_line_times') | (col == 'profile_time') | (col == 'lon') | (col == 'lat') | (col == 'IMERG_preci'):
            continue
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    # end_mem = df.memory_usage().sum() / 1024**2
    # print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    # print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def trunc(values, decs=1):
    return np.trunc(values*10**decs)/(10**decs)


def lat_lon_vectorts_from_tiff(tiff_file):

    """"
    Returns latitude and longitude corresponding to left edge of each pixels of the tiff file.
    
    :param tiff_file: the path of the desired tiff file.
    :return: latitude and longitude vectors.
    """

    with rasterio.open(tiff_file) as src:
        array = src.read(1)
        lat = np.arange(src.transform[5], round(src.transform[5] + (array.shape[0]) * src.transform[4], 2), src.transform[4])
        lon = np.arange(src.transform[2], round(src.transform[2] + (array.shape[1]) * src.transform[0], 2), src.transform[0])
        return lat, lon, array
    

def lat_lon_vectors_from_tiff_rasterio_func(tiff_file):
    """
    Returns 1D latitude and longitude vectors representing the pixel **edges**
    of the raster grid.

    :param tiff_file: Path to the GeoTIFF file.
    :return: (lat_vector, lon_vector), each with length = array.shape + 1
    """
    with rasterio.open(tiff_file) as src:
        array = src.read(1)
        height, width = array.shape
        transform = src.transform

        dx = transform.a  # pixel width
        dy = transform.e  # pixel height (usually negative)

        # x0 = transform.c - dx / 2  # left edge of first column
        # y0 = transform.f - dy / 2  # top edge of first row

        lon = transform.c + dx * np.arange(width + 1)
        lat = transform.f + dy * np.arange(height + 1)

        return lat, lon, array


def index_finder(input_lon, input_lat, lon_bins, lat_bins):

    """""
    Returns the index of the given longitudes and latitudes with respect to the reference longitude and latitude vectors (bins).

    :param inpupt_lon and input_lat: desired 1-D numpy arrays of longitudes and latitudes.
    WARNING!!!!!!!! the input_lon and input_lat should be in ascending and descending order, respectively. WARNING!!!!!!!!
    :param lon_bins and lat_bins: desired 1_D numpy arrays of reference longitude and latitude bins.
    :return: the indices of the bins to which each given input longitude and latitudes belongs.
    """""

    idx = np.digitize(input_lon, lon_bins) - 1
    idy = np.digitize(input_lat, lat_bins, right = True) - 1

    return idx, idy

def index_finder_(input_lon, input_lat, lon_bins, lat_bins):
    """
    Returns the grid indices (idx, idy) for given longitude and latitude values.
    Assumes lon_bins is ascending and lat_bins is descending.
    """
    # For longitude, bins are ascending:
    idx = np.digitize(input_lon, lon_bins) - 1

    # For latitude, reverse the bins to get an ascending order
    lat_bins_asc = lat_bins[::-1]
    # Now, get the index using np.digitize, and then adjust it so that
    # row 0 corresponds to the highest latitude (which was at the beginning of lat_bins).
    idy = len(lat_bins) - np.digitize(input_lat, lat_bins_asc)
    
    return idx, idy

def reference_coordinate_maker(grid_resolution):

    """"
    Returns the reference coordiante matrices (and vectors). It should be noted that the coordinate of the left and upper edge of
    each longitude and latitude pixels are returned, respectively. 

    :param grid_resolution: the desired spatial resolution of the grid.
    """""
    
    x_vec = np.arange(-180, 180, grid_resolution).round(2)
    y_vec = np.arange(90, -90, -grid_resolution).round(2)
    x, y = np.meshgrid(x_vec, y_vec)

    return x_vec, y_vec, x, y

def grid_to_tiff_file(patth, svenme, array2save, grid_resolution):
	# functions saves an array to .tif
	# requires:
			   # patth = the path to save the map
			   # file name for saving the map (add .extension e.g. .tif)
			   # map meta data
			   # array to save

	crss = 'EPSG:4326' # coord reference EPSG:4326  is same as latlong wgs 84
	trns = rasterio.transform.from_origin(-180, 90, grid_resolution, grid_resolution) # creates the geotransform using rasterio module
	array_h, array_w = array2save.shape # check if shape[0],shape[1] corresponds to h,w

	#trns_ = rasterio.transform.from_bounds(-180,-90,180,90,w,h) ; another way to get geotransform

	# make metadat file
	meta_data = {'driver':'GTiff', 'width':array_w, 'height':array_h, 'count':1, 'dtype':'float64',
                'crs':crss, 'transform':trns, 'nodata':np.nan}

	with rasterio.open(os.path.join(patth, svenme), 'w', **meta_data) as mp:
		mp.write(array2save, indexes = 1)

def KGE_fn(sim, obs):
    beta  = np.mean(sim)/np.mean(obs)
    alpha = np.std(sim)/np.std(obs)
    rho   = np.corrcoef(sim, obs)[0,1]
    KGE   = 1 - np.sqrt((1-alpha)**2 + (1-beta)**2 + (1-rho)**2)
    KGEss = (KGE - (1-np.sqrt(2)))/np.sqrt(2)
    return [KGEss, KGE, alpha, beta, rho]

def MSE_Fn(sim,obs):
    MSE = np.mean((sim-obs)**2)
    return MSE

def DML(arr, ML_c, ML_r):

    # Use the forest's predict method on the test data
    predictions = np.empty(arr.shape[0])
    predictions[:] = np.nan

    y_new = ML_c.predict(arr)
    predictions[y_new == 0] = 0

    arr = arr[y_new == 1]
    y_new = ML_r.predict(arr)

    predictions[np.isnan(predictions)] = y_new

    return predictions

def DML_(arr, ML_c, ML_r):

    # Use the forest's predict method on the test data
    predictions = np.empty(arr.shape[0])
    predictions[:] = np.nan

    y_new_c = ML_c.predict(arr)
    predictions[y_new_c == 0] = 0

    arr = arr[y_new_c == 1]
    y_new_r = ML_r.predict(arr)

    predictions[np.isnan(predictions)] = y_new_r

    return predictions, y_new_c


def grid_to_tiff_file_ERA5(patth, svenme, array2save, grid_resolution):
	# functions saves an array to .tif
	# requires:
			   # patth = the path to save the map
			   # file name for saving the map (add .extension e.g. .tif)
			   # map meta data
			   # array to save

	crss = 'EPSG:4326' # coord reference EPSG:4326  is same as latlong wgs 84
	trns = rasterio.transform.from_origin(-180, 90.125, grid_resolution, grid_resolution) # creates the geotransform using rasterio module
	array_h, array_w = array2save.shape # check if shape[0],shape[1] corresponds to h,w

	#trns_ = rasterio.transform.from_bounds(-180,-90,180,90,w,h) ; another way to get geotransform

	# make metadat file
	meta_data = {'driver':'GTiff', 'width':array_w, 'height':array_h, 'count':1, 'dtype':'float64',
                'crs':crss, 'transform':trns, 'nodata':np.nan}

	with rasterio.open(os.path.join(patth, svenme), 'w', **meta_data) as mp:
		mp.write(array2save, indexes = 1)

def grid_to_tiff_file_MERRA2(patth, svenme, array2save, x_grid_resolution, y_grid_resolution):
	# functions saves an array to .tif
	# requires:
			   # patth = the path to save the map
			   # file name for saving the map (add .extension e.g. .tif)
			   # map meta data
			   # array to save

	crss = 'EPSG:4326' # coord reference EPSG:4326  is same as latlong wgs 84
	trns = rasterio.transform.from_origin(-180, 90, x_grid_resolution, y_grid_resolution) # creates the geotransform using rasterio module
	array_h, array_w = array2save.shape # check if shape[0],shape[1] corresponds to h,w

	#trns_ = rasterio.transform.from_bounds(-180,-90,180,90,w,h) ; another way to get geotransform

	# make metadat file
	meta_data = {'driver':'GTiff', 'width':array_w, 'height':array_h, 'count':1, 'dtype':'float64',
                'crs':crss, 'transform':trns, 'nodata':np.nan}

	with rasterio.open(os.path.join(patth, svenme), 'w', **meta_data) as mp:
		mp.write(array2save, indexes = 1)

def IMERG_datetime_NC_files(IMERG_file):

    filename = os.path.basename(IMERG_file)
    dt = datetime.datetime.strptime(filename.split('.')[4][:8], '%Y%m%d')

    return dt

def GMI_datetime(file):

    name_splited = os.path.basename(file).split('.')

    ymd = name_splited[4].split('-')[0]
    yr = ymd[:4]
    st_time = name_splited[4].split('-')[1][1:]
    end_time = name_splited[4].split('-')[2][1:]

    st_dt = datetime.datetime.strptime(ymd + st_time, '%Y%m%d%H%M%S')

    end_dt = datetime.datetime.strptime(ymd + end_time, '%Y%m%d%H%M%S')

    if end_dt < st_dt:
        #files which ending time is few moments after 00:00 pm
        end_dt = end_dt + datetime.timedelta(days=1)

    return st_dt, end_dt, ymd, yr


def IMERG_half_hourly_datetime(file):

    name_splited = os.path.basename(file).split('.')

    time = datetime.datetime.strptime(name_splited[4].split('-')[0] + name_splited[4].split('-')[1][1:], '%Y%m%d%H%M%S')

    return time


def IMERG_half_hourly_reader(file):

    with h5py.File(file, "r") as h5:
        preci = np.flip(h5['/Grid']['precipitation'][:][0].transpose(), axis = 0)
        # preci_16 = np.where(preci == -9999.9, np.nan, preci).astype(np.float16)
        preci = np.where(preci == -9999.9, np.nan, preci)
        # preci = np.where(preci == -9999.9, np.nan, preci).astype(np.float64)

    return preci

def SSMIS_datetime(SSMIS_file):
    
    """"
    Returns datetime index of the introduced SSMIS file.
    
    :param SSMIS_file: SSMIS file
    :return: Datetime index of the SSMIS file
    """
    
    filename = os.path.basename(SSMIS_file)
    dt = datetime.datetime.strptime(filename.split('.')[3][:8], '%Y%m%d')
    return dt


def MHS_start_end_time(file):

    """"
    Returns datetime index of the input MHS file.
    
    :param file: Input MHS file
    :return: Datetime index of the MHS image
    """    

    filename = os.path.basename(file)

    st_dt = datetime.datetime.strptime(filename.split(".")[4].split("-")[0] + filename.split("-S")[1].split("-E")[0], '%Y%m%d%H%M%S')
    end_dt = datetime.datetime.strptime(filename.split(".")[4].split("-")[0] + filename.split("-E")[1].split(".")[0], '%Y%m%d%H%M%S')
    
    if end_dt < st_dt:
        #files which ending time is few moments after 00:00 pm
        end_dt = end_dt + datetime.timedelta(days=1)


    return st_dt, end_dt


def AVHRR_start_end_time(AVHRR_file):

    name_splited = os.path.basename(AVHRR_file).split('.')
    dt = int(name_splited[4][1:] + name_splited[5][1:])

    return dt


def df2grid(df, var_name):

	idx, idy = mf.index_finder(df['lon'], df['lat'], x_vec, y_vec)
	df['idx'] = idx
	df['idy'] = idy

	grid = np.zeros((x.shape[0], x.shape[1])).astype(float)
	grid[:] = np.nan
	grid[df['idy'], df['idx']] = df[var_name]
	return grid



def resample_to_0_1_degree(data, lat, lon, time):
    """
    Resamples 3D data (time, lat, lon) to a 0.1-degree resolution using bilinear interpolation.

    :param data: The 3D data array with dimensions (time, lat, lon)
    :param lat: The latitude array (1D)
    :param lon: The longitude array (1D)
    :param time: The time array (1D)
    :return: Resampled data, resampled latitudes, resampled longitudes
    """

    # Define the target 0.1-degree resolution grid
    # target_lat = np.arange(90, -90.1, -0.1)
    # target_lon = np.arange(-180, 180.1, 0.1)

    target_lat = np.arange(90, -90, -0.1)
    target_lon = np.arange(-180, 180, 0.1)

    # Create the xarray Dataset with the full 3D data
    ds = xr.Dataset(
        {
            "data": (["time", "lat", "lon"], data)
        },
        coords={
            "time": time,
            "lat": lat,
            "lon": lon,
        },
    )

    # Resample the data using bilinear interpolation
    resampled_data = ds.interp(lat=target_lat, lon=target_lon, method="linear")["data"]

    array = resampled_data.values

    # Identify the first and last non-NaN rows and columns
    non_nan_rows = np.where(~np.isnan(array).all(axis=(0, 2)))[0]  # Check along lat and lon dimensions
    non_nan_cols = np.where(~np.isnan(array).all(axis=(0, 1)))[0]  # Check along time and lat dimensions

    # Replace NaN rows by copying the last valid row across all time slices
    if len(non_nan_rows) > 0:
        first_valid_row = non_nan_rows[0]
        array[:, :first_valid_row, :] = array[:, first_valid_row:first_valid_row + 1, :]  # Fill rows above the first valid
        last_valid_row = non_nan_rows[-1]
        array[:, last_valid_row + 1:, :] = array[:, last_valid_row:last_valid_row + 1, :]  # Fill rows below the last valid

    # Replace NaN columns by copying the last valid column across all time slices
    if len(non_nan_cols) > 0:
        first_valid_col = non_nan_cols[0]
        array[:, :, :first_valid_col] = array[:, :, first_valid_col:first_valid_col + 1]  # Fill columns left of the first valid
        last_valid_col = non_nan_cols[-1]
        array[:, :, last_valid_col + 1:] = array[:, :, last_valid_col:last_valid_col + 1]  # Fill columns right of the last valid

    return array, target_lat, target_lon

def resample_to_0_1_degree_NC(ds, lat, lon, time):
    """
    Resamples 3D data (time, lat, lon) to a 0.1-degree resolution using bilinear interpolation.

    :param data: The 3D data array with dimensions (time, lat, lon)
    :param lat: The latitude array (1D)
    :param lon: The longitude array (1D)
    :param time: The time array (1D)
    :return: Resampled data, resampled latitudes, resampled longitudes
    """

    # Define the target 0.1-degree resolution grid
    # target_lat = np.arange(90, -90.1, -0.1)
    # target_lon = np.arange(-180, 180.1, 0.1)

    target_lat = np.arange(90, -90, -0.1)
    target_lon = np.arange(-180, 180, 0.1)

    # Resample the data using bilinear interpolation
    resampled_data = ds.interp(lat=target_lat, lon=target_lon, method="linear")["data"]

    array = resampled_data.values

    # Identify the first and last non-NaN rows and columns
    non_nan_rows = np.where(~np.isnan(array).all(axis=(0, 2)))[0]  # Check along lat and lon dimensions
    non_nan_cols = np.where(~np.isnan(array).all(axis=(0, 1)))[0]  # Check along time and lat dimensions

    # Replace NaN rows by copying the last valid row across all time slices
    if len(non_nan_rows) > 0:
        first_valid_row = non_nan_rows[0]
        array[:, :first_valid_row, :] = array[:, first_valid_row:first_valid_row + 1, :]  # Fill rows above the first valid
        last_valid_row = non_nan_rows[-1]
        array[:, last_valid_row + 1:, :] = array[:, last_valid_row:last_valid_row + 1, :]  # Fill rows below the last valid

    # Replace NaN columns by copying the last valid column across all time slices
    if len(non_nan_cols) > 0:
        first_valid_col = non_nan_cols[0]
        array[:, :, :first_valid_col] = array[:, :, first_valid_col:first_valid_col + 1]  # Fill columns left of the first valid
        last_valid_col = non_nan_cols[-1]
        array[:, :, last_valid_col + 1:] = array[:, :, last_valid_col:last_valid_col + 1]  # Fill columns right of the last valid

    return array, target_lat, target_lon


def grid_to_tiff_file_var_res(patth, svenme, array2save, x_resolution, y_resolution):
    """
    Saves an array to a GeoTIFF file with potentially different resolutions in x and y directions.
    
    :param patth: Path to save the file
    :param svenme: Name of the file (include .tif extension)
    :param array2save: 2D numpy array to save
    :param x_resolution: Resolution in the x direction
    :param y_resolution: Resolution in the y direction
    """
    # Coordinate reference system
    crss = 'EPSG:4326'  # Lat/Lon WGS84

    # Create the geotransform with differing x and y resolutions
    transform = Affine(x_resolution, 0, -180, 0, -y_resolution, 90)

    # Get the array dimensions
    array_h, array_w = array2save.shape

    # Metadata for the GeoTIFF
    meta_data = {
        'driver': 'GTiff',
        'width': array_w,
        'height': array_h,
        'count': 1,
        'dtype': 'float64',
        'crs': crss,
        'transform': transform,
        'nodata': np.nan
    }

    # Save the array as a GeoTIFF
    with rasterio.open(os.path.join(patth, svenme), 'w', **meta_data) as dst:
        dst.write(array2save, indexes=1)

def lat_lon_vectors_from_tiff_rioxarray(tiff_file):
    """
    Returns latitude and longitude corresponding to the left edge of each pixel of the TIFF file.
    
    :param tiff_file: The path of the desired TIFF file.
    :return: Latitude and longitude vectors, which have one more element than the array size.
    """
    # Open the TIFF file as an xarray DataArray
    da = rxr.open_rasterio(tiff_file, masked=True)

    # Extract spatial coordinates
    lon = da.x.values
    lat = da.y.values

    # Adjust lat/lon to represent pixel edges (like in the original function)
    lon = np.append(lon, lon[-1] + (lon[1] - lon[0])) - ((lon[1] - lon[0]) / 2)
    lat = np.append(lat, lat[-1] - (lat[0] - lat[1])) + ((lat[0] - lat[1]) / 2)

    # Extract raster data
    array = da[0].values  # Assuming single-band raster

    return lat, lon, array

def fast_fill_nans(data):
    """Fills NaNs only if they have at least one valid neighbor."""
    
    # Create a mask of NaN values
    nan_mask = np.isnan(data)
    
    # Define a 3x3 convolution kernel to count valid neighbors
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    # Count the number of valid (non-NaN) neighbors
    neighbor_count = convolve2d(~nan_mask, kernel, mode='same', boundary='fill', fillvalue=0)

    # Compute the sum of valid neighbors
    valid_sum = convolve2d(np.nan_to_num(data, nan=0), kernel, mode='same', boundary='fill', fillvalue=0)

    # Compute the mean of valid neighbors where at least one exists
    filled_values = np.where(neighbor_count > 0, valid_sum / neighbor_count, np.nan)

    # Fill only NaN locations
    data[nan_mask] = filled_values[nan_mask]

    return data