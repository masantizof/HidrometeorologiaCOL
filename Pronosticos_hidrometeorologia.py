#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install xarray netCDF4


# In[5]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import Rbf
from matplotlib.path import Path
import geopandas as gpd

# ---------------------------------------------------------
# 1. CONFIGURACIÓN (Rutas y Filtros)
# ---------------------------------------------------------
FOLDER_PATH = r'C:\Users\masan\OneDrive\Documentos\Maestria Meteo\2025-II\HIDROMETEOROLOGIA\Cuenca\Estaciones pluvio'
RUTA_SHP = r'C:\Users\masan\OneDrive\Documentos\Maestria Meteo\2025-II\HIDROMETEOROLOGIA\Cuenca\Datos_Estaciones\Subzonas_Hidrográficas_del_Departamento_del_Tolima.shp'

# Filtro de Cuencaw
NOMBRE_COLUMNA_SHP = 'SUBZONA_HI' 
NOMBRE_CUENCA_OBJETIVO = 'Rio Cucuana' 

# FILTRO TEMPORAL (Ajusta esto para análisis multianual)
FILTRAR_FECHA = True
FECHA_INICIO = '2000-01-01'  # Sugerencia: Usar periodos largos (ej: 1980)
FECHA_FIN = '2020-12-31'     # Sugerencia: Usar periodos largos (ej: 2010)

# Coeficiente de Escorrentía
COEF_ESCORRENTIA = 0.65

NOMBRES_MESES = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']

# Estaciones
estaciones_info = {
    '22075030': {'NOMBRE': 'RIOMANSO', 'LATITUD': 4.2067777780, 'LONGITUD': -75.41558333},
    '22060090': {'NOMBRE': 'OLAYA HERRERA', 'LATITUD': 3.819890, 'LONGITUD': -75.3284},
    '22060070': {'NOMBRE': 'ORTEGA', 'LATITUD': 3.929290, 'LONGITUD': -75.221350},
    '22070030': {'NOMBRE': 'SANTA HELENA', 'LATITUD': 4.1245555560, 'LONGITUD': -75.49952778},
    '21180040': {'NOMBRE': 'ROVIRA 2', 'LATITUD': 4.2425, 'LONGITUD': -75.2425},
    '22070010': {'NOMBRE': 'RONCESVALLES', 'LATITUD': 4.0066388890, 'LONGITUD': -75.607750}
}

# ---------------------------------------------------------
# 2. PROCESAMIENTO GEOMÉTRICO (Solución a interpolación incompleta)
# ---------------------------------------------------------
print("--- Procesando Geometría ---")
if os.path.exists(RUTA_SHP):
    gdf_raw = gpd.read_file(RUTA_SHP)
    if NOMBRE_COLUMNA_SHP in gdf_raw.columns:
        gdf_cuenca = gdf_raw[gdf_raw[NOMBRE_COLUMNA_SHP] == NOMBRE_CUENCA_OBJETIVO].copy()
        if gdf_cuenca.empty: gdf_cuenca = gdf_raw
    else:
        gdf_cuenca = gdf_raw

    # Reproyectar para área
    gdf_metros = gdf_cuenca.to_crs(epsg=3116)
    AREA_M2 = gdf_metros.area.sum()
    gdf_cuenca = gdf_cuenca.to_crs(epsg=4326) # Volver a WGS84

    # --- CORRECCIÓN DE LÍMITES ---
    # Usamos los límites del SHAPEFILE, no de las estaciones, para evitar huecos blancos
    bounds = gdf_cuenca.total_bounds # [minx, miny, maxx, maxy]
    # Agregamos un 10% de margen para asegurar cobertura total
    margin_x = (bounds[2] - bounds[0]) * 0.1
    margin_y = (bounds[3] - bounds[1]) * 0.1
    xlim_min, ylim_min = bounds[0] - margin_x, bounds[1] - margin_y
    xlim_max, ylim_max = bounds[2] + margin_x, bounds[3] + margin_y
else:
    print("❌ Error: No se encontró el SHP.")
    gdf_cuenca = None
    xlim_min, xlim_max, ylim_min, ylim_max = -76, -74, 3, 5
    AREA_M2 = 0

# ---------------------------------------------------------
# 3. CARGA DE DATOS
# ---------------------------------------------------------
print("--- Procesando Datos ---")
csv_files = [f for f in os.listdir(FOLDER_PATH) if f.endswith('.csv')]
data_frames = []

for file in csv_files:
    if '@' in file: station_id = file.split('@')[1].split('.')[0]
    else: station_id = file.replace('.csv', '')

    if station_id not in estaciones_info: continue

    try:
        df = pd.read_csv(os.path.join(FOLDER_PATH, file), skiprows=14, encoding='latin-1', sep=',')
        df.columns = df.columns.str.strip()
        
        if 'Value' in df.columns:
            df['Fecha'] = pd.to_datetime(df['Timestamp (UTC-05:00)'], errors='coerce')
            df['Valor'] = pd.to_numeric(df['Value'], errors='coerce')
            if FILTRAR_FECHA:
                df = df[(df['Fecha'] >= FECHA_INICIO) & (df['Fecha'] <= FECHA_FIN)]
            
            df['Estacion'] = station_id
            df['Mes'] = df['Fecha'].dt.month
            df['Anio'] = df['Fecha'].dt.year
            data_frames.append(df[['Fecha', 'Anio', 'Mes', 'Valor', 'Estacion']])
    except: pass

df_all = pd.concat(data_frames)

# Agrupamos por Estación, Año y Mes (Acumulado Mensual Real)
df_mensual_historico = df_all.groupby(['Estacion', 'Anio', 'Mes'])['Valor'].sum().reset_index()

# Añadimos metadatos
df_mensual_historico['Nombre_Est'] = df_mensual_historico['Estacion'].apply(lambda x: estaciones_info[x]['NOMBRE'])
df_mensual_historico['Nombre_Mes'] = df_mensual_historico['Mes'].apply(lambda x: NOMBRES_MESES[x-1])

# ---------------------------------------------------------
# 4. GRÁFICOS ESTADÍSTICOS MEJORADOS
# ---------------------------------------------------------
print("--- Generando Gráficos Estadísticos ---")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- GRÁFICA 1: BOXPLOT (Dispersión y Valores Atípicos) ---
sns.boxplot(data=df_mensual_historico, x='Nombre_Mes', y='Valor', palette='Blues', ax=axes[0], flierprops={"marker": "x"})
axes[0].set_title('Dispersión de la Precipitación Mensual Multianual\n(Boxplot)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Precipitación (mm)')
axes[0].set_xlabel('')
axes[0].grid(True, linestyle='--', alpha=0.5)

# --- GRÁFICA 2: RÉGIMEN MEDIO (Promedios Mensuales) ---
# Calculamos la media y mediana multianual por mes (promedio de todas las estaciones y años)
df_regimen = df_mensual_historico.groupby('Mes')['Valor'].agg(['mean', 'median']).reset_index()
df_regimen['Nombre_Mes'] = df_regimen['Mes'].apply(lambda x: NOMBRES_MESES[x-1])

# Barras: Promedio
barplot = sns.barplot(data=df_regimen, x='Nombre_Mes', y='mean', color='skyblue', alpha=0.7, ax=axes[1], label='Media (Promedio)')
# Línea: Mediana (Tendencia central más robusta)
axes[1].plot(df_regimen.index, df_regimen['median'], color='darkblue', marker='o', linewidth=2, linestyle='-', label='Mediana')

axes[1].set_title('Régimen de Precipitación Medio Mensual\n(Comportamiento Promedio)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Precipitación (mm)')
axes[1].set_xlabel('')
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.5)

# Anotación de valores sobre las barras
for index, row in df_regimen.iterrows():
    axes[1].text(index, row['mean'] + 5, f"{int(row['mean'])}", color='black', ha="center", fontsize=9)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 5. CÁLCULO DE CAUDALES (Q = P * A * C)
# ---------------------------------------------------------
# Promedio espacial por mes (para el cálculo de Q)
df_climatologia = df_mensual_historico.groupby('Mes')['Valor'].mean().reset_index()
dias_mes = {1:31, 2:28.25, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31} # 28.25 para promedio bisiesto

caudales = []
for i, row in df_climatologia.iterrows():
    mes = int(row['Mes'])
    ppt_mm = row['Valor'] # Precipitación media histórica del mes
    
    ppt_m = ppt_mm / 1000.0
    segundos = dias_mes[mes] * 86400
    q_medio = (ppt_m / segundos) * AREA_M2 * COEF_ESCORRENTIA
    
    caudales.append({'Mes': NOMBRES_MESES[mes-1], 'P_mm': ppt_mm, 'Q_m3s': q_medio})

df_Q = pd.DataFrame(caudales)
print("\n--- Caudales Medios Estimados ---")
print(df_Q)

# ---------------------------------------------------------
# 6. MAPAS MENSUALES (Corregidos)
# ---------------------------------------------------------
print("\n--- Generando Mapas ---")

# Grid basado en el SHAPEFILE (Más amplio para evitar huecos)
grid_x, grid_y = np.meshgrid(np.linspace(xlim_min, xlim_max, 200), 
                             np.linspace(ylim_min, ylim_max, 200))

# Máscara (Clipping)
if gdf_cuenca is not None:
    poly_union = gdf_cuenca.unary_union
    points = np.vstack((grid_x.flatten(), grid_y.flatten())).T
    if poly_union.geom_type == 'MultiPolygon':
        mask = np.zeros(grid_x.shape, dtype=bool).flatten()
        for poly in poly_union.geoms:
            mask = mask | Path(np.array(poly.exterior.coords)).contains_points(points)
    else:
        mask = Path(np.array(poly_union.exterior.coords)).contains_points(points)
    mask = mask.reshape(grid_x.shape)

# Datos para interpolar: Promedio Multianual por Estación y Mes
df_mapas = df_mensual_historico.groupby(['Estacion', 'Mes'])['Valor'].mean().reset_index()
df_mapas['Lat'] = df_mapas['Estacion'].apply(lambda x: estaciones_info[x]['LATITUD'])
df_mapas['Lon'] = df_mapas['Estacion'].apply(lambda x: estaciones_info[x]['LONGITUD'])

fig, axes = plt.subplots(4, 3, figsize=(15, 18))
axes = axes.flatten()

for mes in range(1, 13):
    ax = axes[mes-1]
    nombre_mes = NOMBRES_MESES[mes-1]
    
    # Datos del mes actual
    df_m = df_mapas[df_mapas['Mes'] == mes]
    
    if len(df_m) < 3:
        ax.text(0.5, 0.5, 'Datos insuficientes', ha='center'); continue

    try:
        # Interpolación RBF
        rbf = Rbf(df_m['Lon'], df_m['Lat'], df_m['Valor'], function='thin_plate')
        grid_z = rbf(grid_x, grid_y)
        
        if gdf_cuenca is not None: grid_z[~mask] = np.nan
        
        # Graficar
        niveles = np.linspace(df_m['Valor'].min(), df_m['Valor'].max(), 12)
        cf = ax.contourf(grid_x, grid_y, grid_z, levels=niveles, cmap='Spectral_r', alpha=0.9)
        
        # Shapefile borde
        if gdf_cuenca is not None:
            gdf_cuenca.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.7)
            
        # Puntos
        ax.scatter(df_m['Lon'], df_m['Lat'], c='black', s=10)
        
        # Barra de color individual pequeña
        cbar = plt.colorbar(cf, ax=ax, shrink=0.6)
        cbar.ax.tick_params(labelsize=7)
        
        ax.set_title(nombre_mes, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect('equal')
        
    except Exception as e:
        print(f"Error mes {mes}: {e}")

plt.suptitle(f'Distribución Espacial Media Mensual - {NOMBRE_CUENCA_OBJETIVO}\nPeriodo: {FECHA_INICIO} al {FECHA_FIN}', fontsize=16, y=0.99)
plt.tight_layout()
plt.show()


# In[6]:


import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy.interpolate import Rbf
from matplotlib.path import Path
import os
from datetime import datetime, timedelta

# ---------------------------------------------------------
# 1. CONFIGURACIÓN
# ---------------------------------------------------------
# Ruta del Shapefile (La misma que ya te funcionó)
RUTA_SHP = r'C:\Users\masan\OneDrive\Documentos\Maestria Meteo\2025-II\HIDROMETEOROLOGIA\Cuenca\Datos_Estaciones\Subzonas_Hidrográficas_del_Departamento_del_Tolima.shp'
NOMBRE_COLUMNA_SHP = 'SUBZONA_HI'
NOMBRE_CUENCA_OBJETIVO = 'Rio Cucuana'

# Parámetros Hidrológicos
COEF_ESCORRENTIA = 0.65

# URL del Servidor de Datos (GFS 0.25 grados - "Best Time Series" de Unidata)
# Este enlace siempre apunta a la mejor serie de tiempo disponible combinada
GFS_URL = 'https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/GFS/Global_0p25deg/Best'

# ---------------------------------------------------------
# 2. CARGAR Y PREPARAR LA CUENCA
# ---------------------------------------------------------
print("--- Cargando Geometría de la Cuenca ---")
if os.path.exists(RUTA_SHP):
    gdf_raw = gpd.read_file(RUTA_SHP)
    if NOMBRE_COLUMNA_SHP in gdf_raw.columns:
        gdf_cuenca = gdf_raw[gdf_raw[NOMBRE_COLUMNA_SHP] == NOMBRE_CUENCA_OBJETIVO].copy()
        if gdf_cuenca.empty: gdf_cuenca = gdf_raw
    else:
        gdf_cuenca = gdf_raw

    # Calcular Área
    gdf_metros = gdf_cuenca.to_crs(epsg=3116)
    AREA_M2 = gdf_metros.area.sum()
    gdf_cuenca = gdf_cuenca.to_crs(epsg=4326) # WGS84
    
    # Obtener límites para recortar el GFS (con margen)
    bounds = gdf_cuenca.total_bounds
    min_lon, min_lat, max_lon, max_lat = bounds
    # Margen de 0.5 grados para asegurar que el GFS cubra la zona
    buffer = 0.5
    lat_slice = slice(max_lat + buffer, min_lat - buffer) # GFS suele ir de Norte a Sur
    lon_slice = slice(min_lon - buffer, max_lon + buffer)
    
    print(f"✅ Cuenca: {NOMBRE_CUENCA_OBJETIVO} | Área: {AREA_M2/1e6:.2f} km²")
else:
    print("❌ Error: No se encontró el Shapefile.")
    exit()

# ---------------------------------------------------------
# 3. DESCARGA DE DATOS GFS (OPeNDAP)
# ---------------------------------------------------------
print("\n--- Conectando al Servidor GFS (NOAA/Unidata) ---")
print("Esto puede tardar unos segundos dependiendo de tu internet...")

try:
    # Conectamos remotamente
    ds = xr.open_dataset(GFS_URL)
    
    # 1. Seleccionamos variable: 'Precipitation_rate_surface' (kg/m^2/s)
    # Nota: GFS a veces cambia nombres, pero en THREDDS suele ser este.
    # Convertiremos Tasa (mm/s) a Acumulado (mm) multiplicando por el tiempo.
    
    # 2. Recortamos Espacialmente (Slicing) para no bajar todo el mundo
    # GFS usa longitudes 0-360. Colombia (-75) es 360-75 = 285.
    # Ajuste automático de longitud si es necesario
    if ds.lon.max() > 180:
        lon_slice_gfs = slice((min_lon - buffer) + 360, (max_lon + buffer) + 360)
    else:
        lon_slice_gfs = lon_slice

    # 3. Recortamos Temporalmente (Próximas 72 horas)
    # Buscamos el tiempo inicial más reciente
    t_start = datetime.utcnow()
    t_end = t_start + timedelta(hours=72)
    
    # Descargamos el subset
    subset = ds['Precipitation_rate_surface'].sel(
        lat=lat_slice,
        lon=lon_slice_gfs,
        time=slice(t_start, t_end)
    )
    
    # Cargar en memoria para procesar rápido
    print("Descargando subset de datos...")
    data = subset.load()
    
    # Corregir longitud a -180/180 para el mapa
    if data.lon.max() > 180:
        data.coords['lon'] = (data.coords['lon'] + 180) % 360 - 180
        data = data.sortby('lon')
        
    print("✅ Datos GFS descargados exitosamente.")

except Exception as e:
    print(f"❌ Error descargando GFS: {e}")
    # Datos simulados por si falla la conexión (para que veas el código funcionar)
    print("Generando datos dummy para demostración...")
    lat = np.linspace(min_lat-0.2, max_lat+0.2, 10)
    lon = np.linspace(min_lon-0.2, max_lon+0.2, 10)
    times = pd.date_range(start=datetime.now(), periods=25, freq='3H') # Cada 3h hasta 72h
    data = xr.DataArray(np.random.rand(25, 10, 10) * 0.0001, coords=[times, lat, lon], dims=['time', 'lat', 'lon'])

# ---------------------------------------------------------
# 4. CÁLCULO DE ACUMULADOS (24, 48, 72 HORAS)
# ---------------------------------------------------------
print("\n--- Procesando Pronósticos (24h, 48h, 72h) ---")

horizontes = [24, 48, 72]
resultados = []

for h in horizontes:
    t_fin_h = data.time[0] + np.timedelta64(h, 'h')
    
    # Filtramos las primeras 'h' horas
    slice_h = data.sel(time=slice(data.time[0], t_fin_h))
    
    # --- CONVERSIÓN FÍSICA ---
    # El dato viene en kg/m^2/s (mm/s). Debemos integrar en el tiempo.
    # Método simple: Promedio de tasa * segundos totales
    tasa_media = slice_h.mean(dim='time') # mm/s promedio en el periodo
    segundos = h * 3600
    acumulado_mm = tasa_media * segundos # Total mm en h horas
    
    resultados.append({
        'Hora': h,
        'Grid': acumulado_mm, # Xarray DataArray 2D
        'Max_Ppt': acumulado_mm.max().item(),
        'Mean_Ppt': acumulado_mm.mean().item()
    })

# ---------------------------------------------------------
# 5. CÁLCULO DE CAUDALES
# ---------------------------------------------------------
print("\n--- Estimación de Caudales Pronosticados ---")
print(f"{'Horizonte':<10} | {'Ppt Media (mm)':<15} | {'Caudal Medio (m³/s)':<20}")
print("-" * 50)

for res in resultados:
    ppt_media_mm = res['Mean_Ppt']
    
    # Q = (Ppt_metros / segundos) * Area * C
    ppt_m = ppt_media_mm / 1000.0
    segundos = res['Hora'] * 3600
    
    # Intensidad promedio durante el periodo de pronóstico
    intensidad = ppt_m / segundos
    
    q_pronostico = intensidad * AREA_M2 * COEF_ESCORRENTIA
    
    res['Q_Est'] = q_pronostico
    print(f"{res['Hora']} Horas   | {ppt_media_mm:.2f} mm        | {q_pronostico:.3f} m³/s")

# ---------------------------------------------------------
# 6. GENERACIÓN DE MAPAS
# ---------------------------------------------------------
print("\n--- Generando Mapas de Pronóstico ---")

# Preparamos la malla para interpolar (para suavizar los pixeles grandes del GFS)
grid_x, grid_y = np.meshgrid(np.linspace(min_lon, max_lon, 200), 
                             np.linspace(min_lat, max_lat, 200))

# Máscara de Cuenca
poly_union = gdf_cuenca.unary_union
points = np.vstack((grid_x.flatten(), grid_y.flatten())).T
if poly_union.geom_type == 'MultiPolygon':
    mask = np.zeros(grid_x.shape, dtype=bool).flatten()
    for poly in poly_union.geoms:
        mask = mask | Path(np.array(poly.exterior.coords)).contains_points(points)
else:
    mask = Path(np.array(poly_union.exterior.coords)).contains_points(points)
mask = mask.reshape(grid_x.shape)

# Graficar
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, res in enumerate(resultados):
    ax = axes[i]
    da = res['Grid'] # DataArray del GFS
    
    # Extraer lat/lon/valores del GFS para interpolar
    # Aplanamos los arrays
    gfs_lats, gfs_lons = np.meshgrid(da.lat, da.lon, indexing='ij')
    vals = da.values.flatten()
    lats_flat = gfs_lats.flatten()
    lons_flat = gfs_lons.flatten()
    
    try:
        # Interpolación para suavizar
        rbf = Rbf(lons_flat, lats_flat, vals, function='linear') # Linear es más seguro aquí
        grid_z = rbf(grid_x, grid_y)
        grid_z[~mask] = np.nan # Enmascarar
        
        # Plot
        niveles = np.linspace(0, max(vals.max(), 1), 10) # Evitar error si todo es 0
        cf = ax.contourf(grid_x, grid_y, grid_z, levels=niveles, cmap='YlGnBu')
        
        if gdf_cuenca is not None:
            gdf_cuenca.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)
            
        cbar = plt.colorbar(cf, ax=ax, shrink=0.6)
        cbar.set_label('mm Acumulados')
        
        # Texto
        ax.set_title(f"Pronóstico Acumulado {res['Hora']}h\nQ Est: {res['Q_Est']:.1f} m³/s")
        ax.set_xticks([]); ax.set_yticks([])
        
    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {e}", ha='center')
        print(f"Error graficando {res['Hora']}h: {e}")

plt.suptitle(f"Pronóstico GFS (Precipitación) - {NOMBRE_CUENCA_OBJETIVO}\nInicio: {datetime.now().strftime('%Y-%m-%d %H:00')}", fontsize=16)
plt.tight_layout()
plt.show()


# In[ ]:




