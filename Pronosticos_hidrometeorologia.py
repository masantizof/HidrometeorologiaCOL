"""
balance_hidrico_multianual_v3_abcd.py

Genera el BALANCE HÍDRICO MENSUAL MULTIANUAL + MODELO DE CAUDALES ABCD.

1. Calcula Climatología (Promedio Ene-Dic) de Precipitación.
2. Calcula PET usando Thornthwaite.
3. Ejecuta Balance Clásico (Thornthwaite) para Clasificación Climática.
4. Ejecuta Modelo ABCD para estimar Escorrentía Directa y Flujo Base.
5. Genera reportes (Excel/CSV) y gráficos integrados.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. CONFIGURACIÓN
# ---------------------------------------------------------
FOLDER_PATH = r'C:\Users\masan\OneDrive\Documentos\Maestria Meteo\2025-II\HIDROMETEOROLOGIA\Cuenca\Estaciones pluvio'

# Periodo de Análisis
FECHA_INICIO = '2000-01-01'
FECHA_FIN = '2020-12-31'

# Parámetros Thornthwaite
CAPACIDAD_CAMPO_THORNTHWAITE = 100.0  # mm (Para balance de suelos agrícola)

# Parámetros Modelo ABCD (Teóricos - Ajustar si hay datos de caudal)
# a: Propensión a escorrentía (0-1). 0.98 es usual para cuencas con buena infiltración inicial.
# b: Límite superior de almacenamiento (mm). Similar a capacidad de campo pero suele ser mayor.
# c: Factor de recarga subterránea (0-1). Cuanto va al acuífero.
# d: Constante de recesión flujo base (0-1). Inverso del tiempo de residencia.
PARAMETROS_ABCD = {'a': 0.98, 'b': 250.0, 'c': 0.4, 'd': 0.2}

NOMBRES_MESES = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
KEYS_TEMP = ['ENE', 'FEB', 'MAR', 'ABR', 'MAY', 'JUN', 'JUL', 'AGO', 'SEP', 'OCT', 'NOV', 'DIC']

# Diccionario de Estaciones (Temperaturas)
estaciones_info = {
  "22075030": { "NOMBRE": "RIOMANSO", "LATITUD": 4.206777778, "LONGITUD": -75.41558333,
    "TEMP_MENSUAL": {"ENE": 25.3, "FEB": 25.1, "MAR": 24.6, "ABR": 23.8, "MAY": 22.9, "JUN": 21.5, "JUL": 21.1, "AGO": 21.7, "SEP": 22.4, "OCT": 23.2, "NOV": 22.7, "DIC": 24.1}},
  "22060090": { "NOMBRE": "OLAYA HERRERA", "LATITUD": 3.81989, "LONGITUD": -75.3284,
    "TEMP_MENSUAL": {"ENE": 28.7, "FEB": 28.4, "MAR": 27.9, "ABR": 27.1, "MAY": 26.2, "JUN": 25.0, "JUL": 24.7, "AGO": 25.1, "SEP": 26.0, "OCT": 26.8, "NOV": 25.9, "DIC": 27.5}},
  "22060070": { "NOMBRE": "ORTEGA", "LATITUD": 3.92929, "LONGITUD": -75.22135,
    "TEMP_MENSUAL": {"ENE": 27.9, "FEB": 27.6, "MAR": 27.1, "ABR": 26.4, "MAY": 25.3, "JUN": 24.1, "JUL": 23.8, "AGO": 24.3, "SEP": 25.2, "OCT": 26.1, "NOV": 25.5, "DIC": 26.8}},
  "22070030": { "NOMBRE": "SANTA HELENA", "LATITUD": 4.124555556, "LONGITUD": -75.49952778,
    "TEMP_MENSUAL": {"ENE": 22.8, "FEB": 22.5, "MAR": 22.0, "ABR": 21.3, "MAY": 20.4, "JUN": 19.2, "JUL": 18.7, "AGO": 19.4, "SEP": 20.1, "OCT": 21.0, "NOV": 20.6, "DIC": 21.8}},
  "21180040": { "NOMBRE": "ROVIRA 2", "LATITUD": 4.2425, "LONGITUD": -75.2425,
    "TEMP_MENSUAL": {"ENE": 23.4, "FEB": 23.1, "MAR": 22.7, "ABR": 21.8, "MAY": 21.0, "JUN": 19.9, "JUL": 19.4, "AGO": 20.0, "SEP": 20.7, "OCT": 21.5, "NOV": 21.2, "DIC": 22.6}},
  "22070010": { "NOMBRE": "RONCESVALLES", "LATITUD": 4.006638889, "LONGITUD": -75.60775,
    "TEMP_MENSUAL": {"ENE": 18.4, "FEB": 18.1, "MAR": 17.7, "ABR": 17.1, "MAY": 16.4, "JUN": 15.8, "JUL": 15.6, "AGO": 16.0, "SEP": 16.5, "OCT": 17.2, "NOV": 17.0, "DIC": 17.9}}
}

# ---------------------------------------------------------
# 2. FUNCIONES DE LECTURA Y UTILIDADES
# ---------------------------------------------------------
def leer_y_promediar_precipitacion(path):
    """
    Lee histórico, filtra fecha y calcula promedio mensual multianual.
    """
    try:
        try: df = pd.read_csv(path, encoding='utf-8', skiprows=14)
        except: df = pd.read_csv(path, encoding='latin1', skiprows=14)
        
        if df.empty or len(df.columns) < 2:
            df = pd.read_csv(path, encoding='latin1')

        df.columns = df.columns.str.strip()
        
        col_fecha = next((c for c in df.columns if 'fecha' in c.lower() or 'date' in c.lower() or 'timestamp' in c.lower()), None)
        col_valor = next((c for c in df.columns if any(x in c.lower() for x in ['precip', 'valor', 'value', 'rain', 'lluvia'])), None)

        if not col_fecha or not col_valor: return None

        df[col_fecha] = pd.to_datetime(df[col_fecha], errors='coerce')
        df[col_valor] = pd.to_numeric(df[col_valor], errors='coerce')
        df = df.dropna(subset=[col_fecha, col_valor])
        
        df = df[(df[col_fecha] >= FECHA_INICIO) & (df[col_fecha] <= FECHA_FIN)]
        
        df['anio'] = df[col_fecha].dt.year
        df['mes'] = df[col_fecha].dt.month
        
        # Suma mensual por año y luego promedio de esos sumatorios
        mensual_por_anio = df.groupby(['anio', 'mes'])[col_valor].sum().reset_index()
        promedio_climatologico = mensual_por_anio.groupby('mes')[col_valor].mean().sort_index()
        promedio_climatologico = promedio_climatologico.reindex(range(1,13), fill_value=0)
        
        return promedio_climatologico.values
    except Exception as e:
        print(f"  [!] Error leyendo {path.name}: {e}")
        return None

def calcular_factor_latitud(latitud, mes_num):
    dias_mes = [0, 31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    phi = np.radians(latitud)
    delta = 0.409 * np.sin(2 * np.pi * mes_num / 12 - 1.39)
    try:
        omega = np.arccos(-np.tan(phi) * np.tan(delta))
        N = 24 / np.pi * omega
    except: N = 12
    return (N / 12) * (dias_mes[mes_num] / 30)

# ---------------------------------------------------------
# 3. MODELOS HIDROLÓGICOS
# ---------------------------------------------------------

def ejecutar_thornthwaite_y_abcd(P_med, T_med, latitud, cap_th, params_abcd):
    """
    Ejecuta ambos modelos sobre el año promedio climatológico.
    """
    # --- A. PREPARAR DATOS Y PET ---
    df = pd.DataFrame({'Mes_Num': range(1,13), 'Mes': NOMBRES_MESES, 'P_mm': P_med, 'T_C': T_med})
    
    I_anual = np.sum((np.array(T_med) / 5) ** 1.514)
    a_th = (6.75e-7 * I_anual**3) - (7.71e-5 * I_anual**2) + (1.792e-2 * I_anual) + 0.49239
    
    df['Et_raw'] = 16 * ((10 * df['T_C']) / I_anual) ** a_th
    df['F_Lat'] = df['Mes_Num'].apply(lambda x: calcular_factor_latitud(latitud, x))
    df['PET_mm'] = df['Et_raw'] * df['F_Lat']
    
    # --- B. BALANCE THORNTHWAITE (Clásico para Clasificación) ---
    alm_prev = cap_th
    # Calentamiento TH
    for _ in range(3):
        for idx in range(12):
            diff = df.loc[idx, 'P_mm'] - df.loc[idx, 'PET_mm']
            nuevo = min(max(alm_prev + diff, 0), cap_th)
            alm_prev = nuevo
            
    # Ejecución TH
    th_alm, th_etr, th_exc, th_def = [], [], [], []
    for i in range(12):
        diff = df.loc[i, 'P_mm'] - df.loc[i, 'PET_mm']
        nuevo = min(max(alm_prev + diff, 0), cap_th)
        da = nuevo - alm_prev
        
        if diff >= 0:
            etr = df.loc[i, 'PET_mm']
            exc = diff - da
            dft = 0
        else:
            etr = df.loc[i, 'P_mm'] + abs(da)
            exc = 0
            dft = df.loc[i, 'PET_mm'] - etr
            
        th_alm.append(nuevo)
        th_etr.append(etr)
        th_exc.append(exc)
        th_def.append(dft)
        alm_prev = nuevo
        
    df['TH_Alm'] = th_alm
    df['TH_Etr'] = th_etr
    df['TH_Def'] = th_def
    df['TH_Exc'] = th_exc # Exceso climático (no necesariamente caudal)

    # --- C. MODELO ABCD (Para Caudales) ---
    a = params_abcd['a']
    b = params_abcd['b']
    c = params_abcd['c']
    d = params_abcd['d']
    
    # Estados Iniciales (Arbitrarios, se estabilizan con warm-up)
    S_abcd = b / 2  # Almacenamiento Suelo
    G_abcd = 100    # Almacenamiento Acuífero
    
    # Calentamiento ABCD (3 años)
    for _ in range(3):
        for i in range(12):
            P = df.loc[i, 'P_mm']
            PET = df.loc[i, 'PET_mm']
            
            # Agua disponible total
            W = P + S_abcd
            
            # Oportunidad de pérdida (Y) - Fórmula no lineal de Thomas
            # Y representa (Etr + S_nuevo)
            term1 = (W + b) / (2 * a)
            term2 = (W * b) / a
            # Evitar raices negativas en casos extremos
            disc = term1**2 - term2
            if disc < 0: Y = W
            else: Y = term1 - np.sqrt(disc)
            
            # Nuevo almacenamiento suelo
            S_new = Y * np.exp(-PET / b)
            
            # Agua disponible para escorrentía (Detention)
            # Lo que no se evapora ni se queda en el suelo, baja.
            avail = W - Y 
            
            # Partición (Recharge vs Direct Runoff)
            rech = c * avail
            
            # Acuífero
            G_new = (rech + G_abcd) / (1 + d)
            
            # Actualizar
            S_abcd = S_new
            G_abcd = G_new

    # Ejecución ABCD (Guardando datos)
    abcd_S, abcd_G = [], []
    q_direct, q_base, q_total = [], [], []
    abcd_etr = []
    
    for i in range(12):
        P = df.loc[i, 'P_mm']
        PET = df.loc[i, 'PET_mm']
        
        W = P + S_abcd
        
        term1 = (W + b) / (2 * a)
        term2 = (W * b) / a
        disc = term1**2 - term2
        if disc < 0: Y = W
        else: Y = term1 - np.sqrt(disc)
        
        S_new = Y * np.exp(-PET / b)
        E_real = Y - S_new # Evapotranspiración Real ABCD
        
        avail = W - Y # Agua sobrante del tanque superior
        
        Q_dir = (1 - c) * avail # Escorrentía Directa
        Rech = c * avail        # Recarga
        
        # Tanque inferior
        G_new = (Rech + G_abcd) / (1 + d)
        Q_bas = d * G_new       # Flujo Base
        
        # Guardar
        abcd_S.append(S_new)
        abcd_G.append(G_new)
        q_direct.append(Q_dir)
        q_base.append(Q_bas)
        q_total.append(Q_dir + Q_bas)
        abcd_etr.append(E_real)
        
        # Actualizar
        S_abcd = S_new
        G_abcd = G_new
        
    df['ABCD_S'] = abcd_S
    df['ABCD_G'] = abcd_G
    df['Q_Directo'] = q_direct
    df['Q_Base'] = q_base
    df['Q_Total'] = q_total
    
    return df

def graficar_integrado(df, nombre_estacion, im_val, output_path):
    """Gráfica que combina Balance Climático (Áreas) y Caudal Simulado (Línea)."""
    fig, ax = plt.subplots(figsize=(11, 6))
    x = range(1, 13)
    
    # 1. Fondo: Balance Thornthwaite (Exceso/Déficit Climático)
    ax.fill_between(x, df['P_mm'], df['PET_mm'], where=(df['P_mm'] > df['PET_mm']),
                    interpolate=True, color='dodgerblue', alpha=0.2, label='Exceso Climático (Thornthwaite)')
    ax.fill_between(x, df['P_mm'], df['PET_mm'], where=(df['P_mm'] <= df['PET_mm']),
                    interpolate=True, color='crimson', alpha=0.2, label='Déficit Climático')
    
    # 2. Líneas Principales
    ax.plot(x, df['P_mm'], marker='o', color='royalblue', linewidth=2, label='Precipitación (P)')
    ax.plot(x, df['PET_mm'], linestyle='--', color='firebrick', label='PET (Thornthwaite)')
    
    # 3. Caudal ABCD
    ax.plot(x, df['Q_Total'], color='black', linewidth=2.5, linestyle='-', marker='s', markersize=5, label='Caudal Simulado (Q Total ABCD)')
    ax.plot(x, df['Q_Base'], color='gray', linewidth=1.5, linestyle=':', label='Flujo Base (Acuífero)')
    
    clasif = "Húmedo" if im_val > 0 else "Seco"
    ax.set_title(f"Balance Hídrico y Caudales (ABCD)\nEstación: {nombre_estacion} (Im={im_val:.1f})", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(NOMBRES_MESES)
    ax.set_ylabel("Lámina de Agua (mm)")
    
    # Leyenda fuera para no tapar
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

# ---------------------------------------------------------
# 4. PROCESO PRINCIPAL
# ---------------------------------------------------------
def main():
    folder = Path(FOLDER_PATH)
    if not folder.exists(): return print("Error: Carpeta no encontrada.")

    out_dir = folder / "Reporte_Multianual_ABCD"
    out_dir.mkdir(exist_ok=True)
    
    print("--- GENERANDO REPORTE INTEGRADO (Thornthwaite + ABCD) ---")
    
    archivos_todos = []
    for ext in ('*.csv','*.CSV','*.xlsx','*.xls'):
        archivos_todos.extend(list(folder.rglob(ext)))
    archivos_map = {f.name.lower(): f for f in archivos_todos}
    
    # Excel Consolidado
    excel_path = out_dir / "Consolidado_Hidrologico.xlsx"
    try: writer_excel = pd.ExcelWriter(excel_path)
    except: writer_excel = None
    
    count = 0
    for codigo, info in estaciones_info.items():
        nombre = info['NOMBRE']
        print(f"Procesando: {nombre}...")
        
        f_path = next((v for k, v in archivos_map.items() if codigo in k or nombre.lower() in k), None)
        if not f_path:
            print("  [X] Archivo no encontrado.")
            continue
            
        P_med = leer_y_promediar_precipitacion(f_path)
        if P_med is None: continue
        
        T_med = [info['TEMP_MENSUAL'][k] for k in KEYS_TEMP]
        
        # EJECUTAR MODELOS
        df_res = ejecutar_thornthwaite_y_abcd(
            P_med, T_med, info['LATITUD'], 
            CAPACIDAD_CAMPO_THORNTHWAITE, PARAMETROS_ABCD
        )
        
        # Indices Climáticos (Usando salidas de Thornthwaite para clasificación)
        sum_exc = df_res['TH_Exc'].sum()
        sum_def = df_res['TH_Def'].sum()
        sum_pet = df_res['PET_mm'].sum()
        im_val = (100 * sum_exc - 60 * sum_def) / sum_pet if sum_pet > 0 else 0
        
        # Guardar CSV
        df_res.round(2).to_csv(out_dir / f"Datos_{nombre}.csv", index=False)
        
        # Generar Gráfica Integrada
        graficar_integrado(df_res, nombre, im_val, out_dir / f"Grafica_{nombre}.png")
        
        # Guardar Excel
        if writer_excel:
            try:
                sheet = nombre.replace('[','').replace(']','')[:30]
                df_res.round(2).to_excel(writer_excel, sheet_name=sheet, index=False)
            except: pass
            
        print(f"  -> OK. Q_Total Anual Est.: {df_res['Q_Total'].sum():.1f} mm")
        count += 1
        
    if writer_excel:
        writer_excel.close()
        print(f"\nResultados guardados en: {out_dir}")

if __name__ == "__main__":
    main()
    
