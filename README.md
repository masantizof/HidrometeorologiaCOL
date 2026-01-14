# 🌦️ Hydrometeorological Forecasting Engine

![Python](https://img.shields.io/badge/Python-Data%20Science-blue) ![Model](https://img.shields.io/badge/Model-GFS-yellow) ![Domain](https://img.shields.io/badge/Domain-Meteorology-teal)

## 📋 Project Overview
A computational framework developed to bridge the gap between Global Climate Models (GCMs) and local hydrological needs. This engine processes data from the **Global Forecast System (GFS)** to generate localized precipitation and temperature forecasts for the **Tolima River Basin**.

It demonstrates the application of physical principles and spatial statistics to solve water resource management problems.

## 🧪 Scientific & Technical Approach
1.  **Data Ingestion:** Automated parsing of multidimensional meteorological data (NetCDF/GRIB formats) using `xarray`.
2.  **Spatial Downscaling:** Implementation of **Radial Basis Functions (Rbf)** to interpolate coarse grid points (0.25°) to a high-resolution local surface.
3.  **Basin Analysis:** Integration with `geopandas` to mask and extract statistics specifically for the watershed's shapefile.

## 🛠️ Tech Stack
* **Core:** Python, Numpy, Pandas.
* **Geospatial:** `xarray`, `netCDF4`, `geopandas`, `scipy.interpolate`.
* **Visualization:** `matplotlib`, `seaborn` for generating forecast maps.

## 📊 Sample Output
* *Precipitation accumulation maps (24h - 168h).*
* *Temperature distribution profiles adjusted for local topography.*
