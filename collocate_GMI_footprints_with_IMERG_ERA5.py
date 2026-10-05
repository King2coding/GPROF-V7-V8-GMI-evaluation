"""
Footprint-level GMI matchup with nearest IMERG and ERA5 values.

This keeps GMI at native footprint level. For each valid GMI footprint, it finds
the nearest IMERG half-hourly field and nearest ERA5 hourly field in time, then
samples the nearest grid cell in location.
"""
import argparse
import datetime
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
#
import h5py
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset

import my_functions as mf

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


GMI_V8_DIR = '/scratch/kkumah/GPM_GMI/V8'
GMI_V7_DIR = '/scratch/kkumah/GPM_GMI/V7'
IMERG_DIR = '/ra1/pubdat/AVHRR_CloudSat_proj/IMERG/IMERGV7/Data_V7_halfhourly_20162018/'
ERA5_DIR = '/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_0.25deg/'
DEFAULT_OUTPUT_DIR = '/home/omidzandi/GMI_footprint_IMERG_ERA5_matchups'
IMERG_META = None
ERA5_META = None


def _progress(iterable, total):
    if tqdm is not None:
        return tqdm(iterable, total=total, desc='GMI orbits', unit='orbit')
    return iterable


def _orbit_id_from_gmi_file(gmi_file):
    parts = os.path.basename(gmi_file).split('.')
    if len(parts) >= 6:
        return f"{parts[4]}.{parts[5]}"
    return os.path.basename(gmi_file)


def _read_gmi_footprints_h5(gmi_file, version='v8', lat_abs_min=None):
    with h5py.File(gmi_file, 'r') as f:
        s1 = f['S1']
        precip_raw = s1['surfacePrecipitation'][:]
        fill_value = s1['surfacePrecipitation'].attrs.get('_FillValue', [-9999.9])[0]
        lat = s1['Latitude'][:]
        lon = s1['Longitude'][:]

        scan_time = s1['ScanTime']
        time_df = pd.DataFrame({
            'year': np.array([str(i) for i in scan_time['Year'][:]]),
            'month': np.array(['{:02d}'.format(i) for i in scan_time['Month'][:]]),
            'day': np.array(['{:02d}'.format(i) for i in scan_time['DayOfMonth'][:]]),
            'hour': np.array(['{:02d}'.format(i) for i in scan_time['Hour'][:]]),
            'minute': np.array(['{:02d}'.format(i) for i in scan_time['Minute'][:]]),
            'second': np.array(['{:02d}'.format(i) for i in scan_time['Second'][:]]),
        })

    ymdhms = time_df['year'] + time_df['month'] + time_df['day'] + time_df['hour'] + time_df['minute'] + time_df['second']
    scan_datetimes = np.array([datetime.datetime.strptime(i, '%Y%m%d%H%M%S') for i in ymdhms])
    scan_stamps = np.array([i.timestamp() for i in scan_datetimes], dtype='float64')
    time_grid = np.tile(scan_stamps.reshape(-1, 1), precip_raw.shape[1])

    scan_idx = np.tile(np.arange(precip_raw.shape[0]).reshape(-1, 1), precip_raw.shape[1])
    pixel_idx = np.tile(np.arange(precip_raw.shape[1]).reshape(1, -1), (precip_raw.shape[0], 1))

    mask = np.isfinite(lat) & np.isfinite(lon) & (precip_raw != fill_value)
    if lat_abs_min is not None:
        mask &= np.abs(lat) >= lat_abs_min

    if not np.any(mask):
        return pd.DataFrame()

    return pd.DataFrame({
        'gmi_orbit': _orbit_id_from_gmi_file(gmi_file),
        'gmi_version': version,
        'scan_idx': scan_idx[mask].astype('uint16'),
        'pixel_idx': pixel_idx[mask].astype('uint16'),
        'gmi_time': pd.to_datetime(time_grid[mask], unit='s'),
        'gmi_time_stamp': time_grid[mask].astype('float64'),
        'lon': lon[mask].astype('float32'),
        'lat': lat[mask].astype('float32'),
        'GMI_preci': precip_raw[mask].astype('float32'),
    })


def _nearest_indices(values, grid):
    values = np.asarray(values)
    grid = np.asarray(grid)
    order = np.argsort(grid)
    sorted_grid = grid[order]
    pos = np.searchsorted(sorted_grid, values)
    pos0 = np.clip(pos - 1, 0, sorted_grid.size - 1)
    pos1 = np.clip(pos, 0, sorted_grid.size - 1)
    use1 = np.abs(sorted_grid[pos1] - values) < np.abs(sorted_grid[pos0] - values)
    nearest_pos = np.where(use1, pos1, pos0)
    return order[nearest_pos]


def _nearest_time_indices(stamps, reference_stamps):
    return _nearest_indices(stamps, reference_stamps)


def _load_imerg_metadata():
    files = np.array(mf.dir_files(IMERG_DIR, 'HDF5'))
    times = np.array([mf.IMERG_half_hourly_datetime(i) for i in files])
    stamps = np.array([i.timestamp() for i in times], dtype='float64')
    with h5py.File(files[0], 'r') as h5:
        lon = np.around(h5['/Grid']['lon'][:], decimals=2) - 0.05
        lat = np.around(h5['/Grid']['lat'][:][::-1], decimals=2) + 0.05
    return files, times, stamps, lon, lat


def _read_imerg_precip(imerg_file):
    with h5py.File(imerg_file, 'r') as h5:
        arr = np.flip(h5['Grid/precipitation'][:][0].transpose(), axis=0)
    return np.where(arr == -9999.9, np.nan, arr).astype('float32')


def _sample_imerg(df, imerg_files, imerg_times, imerg_stamps, imerg_lon, imerg_lat):
    time_idx = _nearest_time_indices(df['gmi_time_stamp'].values, imerg_stamps)
    x_idx = _nearest_indices(df['lon'].values, imerg_lon)
    y_idx = _nearest_indices(df['lat'].values, imerg_lat)

    vals = np.full(df.shape[0], np.nan, dtype='float32')
    for idx in np.unique(time_idx):
        rows = np.where(time_idx == idx)[0]
        arr = _read_imerg_precip(imerg_files[idx])
        vals[rows] = arr[y_idx[rows], x_idx[rows]]

    df['IMERG_preci'] = vals
    df['IMERG_time'] = pd.to_datetime(imerg_stamps[time_idx], unit='s')
    return df


def _load_era5_metadata():
    files = np.array(mf.dir_files(ERA5_DIR, 'nc')[:-1])
    era5_yrs = [int(os.path.basename(i).split('_')[2][:4]) for i in files]
    file_by_year = dict(zip(era5_yrs, files))

    ymdh_by_year = {}
    index_by_year_ymdh = {}
    reference_time = datetime.datetime.strptime('1900-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    for file in files:
        year = int(os.path.basename(file)[13:17])
        with Dataset(file) as nc:
            converted_time = [reference_time + datetime.timedelta(hours=int(i)) for i in nc['time'][:]]
        ymdh_list = ['{:02d}'.format(i.year) + '{:02d}'.format(i.month) + '{:02d}'.format(i.day) + '{:02d}'.format(i.hour) for i in converted_time]
        ymdh_by_year[year] = ymdh_list
        index_by_year_ymdh[year] = {ymdh: i for i, ymdh in enumerate(ymdh_list)}

    era5_lat = np.arange(90.125, -90.25, -0.25).round(3)
    era5_lon = np.arange(-180, 180.25, 0.25).round(2)
    era5_lon_org = np.hstack((np.arange(0, 180, 0.25), np.arange(-180, 0, 0.25))).round(2)
    lon_mask = era5_lon_org.argsort()
    return file_by_year, ymdh_by_year, index_by_year_ymdh, era5_lon, era5_lat, lon_mask


def _sample_era5(df, era5_meta):
    file_by_year, ymdh_by_year, index_by_year_ymdh, era5_lon, era5_lat, lon_mask = era5_meta

    era5_time = df['gmi_time'].dt.round('h') + pd.Timedelta(hours=1)
    era5_ymdh = (
        era5_time.dt.year.astype(str).str.zfill(4)
        + era5_time.dt.month.astype(str).str.zfill(2)
        + era5_time.dt.day.astype(str).str.zfill(2)
        + era5_time.dt.hour.astype(str).str.zfill(2)
    )

    idx, idy = mf.index_finder(df['lon'].values, df['lat'].values, era5_lon, era5_lat)
    valid = (
        (idx != -1) & (idy != -1) &
        (idx != era5_lon.shape[0] - 1) & (idy != era5_lat.shape[0] - 1)
    )

    vals = np.full(df.shape[0], np.nan, dtype='float32')
    keys = np.array([f"{t.year}_{ymdh}" for t, ymdh in zip(era5_time, era5_ymdh)], dtype=object)
    for key in np.unique(keys):
        rows = np.where((keys == key) & valid)[0]
        if rows.size == 0:
            continue
        year, ymdh = key.split('_')
        year = int(year)
        if year not in index_by_year_ymdh or ymdh not in index_by_year_ymdh[year]:
            continue
        time_index = index_by_year_ymdh[year][ymdh]
        with Dataset(file_by_year[year]) as nc:
            tp = np.where(nc['tp'][time_index, :, :].mask, np.nan, nc['tp'][time_index, :, :].data) * 1000
        tp = tp[:, lon_mask]
        tp = np.where(tp < 0, 0, tp).astype('float32')
        vals[rows] = tp[idy[rows], idx[rows]]

    df['ERA5_tp'] = vals
    df['ERA5_time'] = era5_time
    return df


def _save_dataframe(df, output_file):
    ext = os.path.splitext(output_file)[1].lower()
    if ext in ['.pkl', '.pickle']:
        df.to_pickle(output_file)
    elif ext == '.parquet':
        df.to_parquet(output_file, index=False)
    elif ext == '.csv':
        df.to_csv(output_file, index=False)
    else:
        raise ValueError('Output extension must be .pkl, .pickle, .parquet, or .csv')


def _save_footprint_nc(df, output_file, gmi_file, version):
    nscan = int(df['scan_idx'].max()) + 1
    npixel = int(df['pixel_idx'].max()) + 1

    arrays = {}
    for var in ['GMI_preci', 'IMERG_preci', 'ERA5_tp', 'lat', 'lon']:
        arr = np.full((nscan, npixel), np.nan, dtype='float32')
        arr[df['scan_idx'].values, df['pixel_idx'].values] = df[var].values.astype('float32')
        arrays[var] = arr

    gmi_time = np.full(nscan, np.nan, dtype='float64')
    era5_time = np.full(nscan, np.nan, dtype='float64')
    imerg_time = np.full(nscan, np.nan, dtype='float64')
    scan_times = df.groupby('scan_idx', observed=True)[['gmi_time', 'IMERG_time', 'ERA5_time']].first()
    scan_idx = scan_times.index.values.astype(int)
    gmi_time[scan_idx] = scan_times['gmi_time'].astype('int64').values / 1e9
    imerg_time[scan_idx] = scan_times['IMERG_time'].astype('int64').values / 1e9
    era5_time[scan_idx] = scan_times['ERA5_time'].astype('int64').values / 1e9

    with Dataset(output_file, 'w', format='NETCDF4') as nc:
        nc.createDimension('scan', nscan)
        nc.createDimension('pixel', npixel)

        scan_var = nc.createVariable('scan', 'u2', ('scan',))
        pixel_var = nc.createVariable('pixel', 'u2', ('pixel',))
        scan_var[:] = np.arange(nscan, dtype='uint16')
        pixel_var[:] = np.arange(npixel, dtype='uint16')

        for var in ['GMI_preci', 'IMERG_preci', 'ERA5_tp', 'lat', 'lon']:
            out = nc.createVariable(var, 'f4', ('scan', 'pixel'), zlib=True, complevel=4, fill_value=np.nan)
            out[:] = arrays[var]
            if var in ['GMI_preci', 'IMERG_preci', 'ERA5_tp']:
                out.units = 'mm/hr'
            elif var == 'lat':
                out.units = 'degrees_north'
            elif var == 'lon':
                out.units = 'degrees_east'

        for var, values in [('gmi_time', gmi_time), ('IMERG_time', imerg_time), ('ERA5_time', era5_time)]:
            out = nc.createVariable(var, 'f8', ('scan',), zlib=True, complevel=4, fill_value=np.nan)
            out[:] = values
            out.units = 'seconds since 1970-01-01 00:00:00'

        nc.GMI_input_file = os.path.basename(gmi_file)
        nc.GMI_version = version
        nc.description = 'Native GMI footprint-level matchup with nearest IMERG and ERA5 values.'


def collocate_one_gmi_file(gmi_file, version, output_dir, lat_abs_min=None, output_format='nc'):
    global IMERG_META, ERA5_META
    if IMERG_META is None:
        IMERG_META = _load_imerg_metadata()
    if ERA5_META is None:
        ERA5_META = _load_era5_metadata()

    imerg_meta = IMERG_META
    era5_meta = ERA5_META
    imerg_files, imerg_times, imerg_stamps, imerg_lon, imerg_lat = imerg_meta

    df = _read_gmi_footprints_h5(gmi_file, version=version, lat_abs_min=lat_abs_min)
    if df.empty:
        return None, 0

    df = _sample_imerg(df, imerg_files, imerg_times, imerg_stamps, imerg_lon, imerg_lat)
    df = _sample_era5(df, era5_meta)

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, os.path.basename(gmi_file).replace('.HDF5', '').replace('.nc', ''))
    output_files = []

    if output_format in ['nc', 'both']:
        nc_file = base + '.nc'
        _save_footprint_nc(df, nc_file, gmi_file, version)
        output_files.append(nc_file)

    if output_format in ['pkl', 'pickle', 'parquet', 'csv', 'both']:
        df_out = df.drop(columns=['gmi_time_stamp'])
        if output_format == 'both':
            df_file = base + '.pkl'
        else:
            ext = 'pkl' if output_format in ['pkl', 'pickle'] else output_format
            df_file = base + f'.{ext}'
        _save_dataframe(df_out, df_file)
        output_files.append(df_file)

    return output_files, df.shape[0]


def _list_gmi_files(version):
    if version == 'v8':
        return np.array(sorted(glob(os.path.join(GMI_V8_DIR, '*.nc'))))
    if version == 'v7':
        return np.array(sorted(glob(os.path.join(GMI_V7_DIR, '*.HDF5'))))
    raise ValueError('Only one version at a time is supported in this script.')


def main():
    parser = argparse.ArgumentParser(description='Footprint-level GMI matchup with nearest IMERG and ERA5.')
    parser.add_argument('--gmi-version', choices=['v8', 'v7'], default='v8')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--year-start', type=int, default=2016)
    parser.add_argument('--year-end', type=int, default=2017)
    parser.add_argument('--max-files', type=int, default=None)
    parser.add_argument('--lat-abs-min', type=float, default=None, help='Optional absolute latitude threshold.')
    parser.add_argument('--output-format', choices=['nc', 'pkl', 'parquet', 'csv', 'both'], default='nc')
    args = parser.parse_args()

    files = _list_gmi_files(args.gmi_version)
    years = np.array([int(os.path.basename(i).split('.')[4].split('-')[0][:4]) for i in files])
    files = files[(years >= args.year_start) & (years <= args.year_end)]
    if args.max_files is not None:
        files = files[:args.max_files]

    output_dir = os.path.join(args.output_dir, args.gmi_version)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing {files.size} GMI {args.gmi_version.upper()} files")
    print(f"Output directory: {output_dir}")

    if files.size == 0:
        return

    if args.workers == 1:
        for file in _progress(files, total=files.size):
            try:
                output_files, n = collocate_one_gmi_file(
                    file,
                    args.gmi_version,
                    output_dir,
                    lat_abs_min=args.lat_abs_min,
                    output_format=args.output_format,
                )
                print(f"{os.path.basename(file)} rows={n} output={output_files}")
            except Exception as exc:
                print(f"{os.path.basename(file)} failed: {exc}")
        return

    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futures = {
            exe.submit(
                collocate_one_gmi_file,
                file,
                args.gmi_version,
                output_dir,
                args.lat_abs_min,
                args.output_format,
            ): file
            for file in files
        }
        for fut in _progress(as_completed(futures), total=len(futures)):
            file = futures[fut]
            try:
                output_files, n = fut.result()
                print(f"{os.path.basename(file)} rows={n} output={output_files}")
            except Exception as exc:
                print(f"{os.path.basename(file)} failed: {exc}")


if __name__ == '__main__':
    main()
