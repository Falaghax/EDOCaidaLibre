import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar

# extracción de la data del globo y de la pelota
globoDataRaw = pd.read_csv("Tracker_globo_Raw_Data.txt", sep=";", skiprows=2, names=["t", "y", "v", "a"], usecols=[0,1,2,3])
pelotaDataRaw = pd.read_csv("Tracker_pelota_Raw_Data.txt", sep=";", skiprows=2, names=["t", "y", "v", "a"], usecols=[0,1,2,3])
# quiero quitar los primeros y ultimos datos porque son basicamente basura y no me sirven mucho
globoDataRaw.drop(index=globoDataRaw.index[0], axis=0, inplace=True)
globoDataRaw.drop(index=globoDataRaw.index[-1], axis=0, inplace=True)

pelotaDataRaw.drop(index=pelotaDataRaw.index[0], axis=0, inplace=True)
pelotaDataRaw.drop(index=pelotaDataRaw.index[-1], axis=0, inplace=True)
######################################################################

# definicion de la EDO para caida libre v'=g-(k/m)v
def f(t, v, m, k):
    return 9.81-(k/m)*v

# funcion para el modelo de runge-kutta
def rungeKutta4(f,t,v,h,m,k):
    k1 = f(t,v,m,k)
    k2 = f(t+h/2, v+h*k1/2,m,k)
    k3 = f(t+h/2, v+h*k2/2,m,k)
    k4 = f(t+h, v+h*k3,m,k)
    return v+(h/6) * (k1+2*k2+2*k3+k4)

######################################################################
#                 AJUSTE AUTOMÁTICO DE k Y GRÁFICAS
#                 (A partir de aqui, el código, no es de mi autoría,
#                 fue asistido por parte de la herramienta Claude Code)
######################################################################

# Datos limpios (sin filas con v=NaN) para el optimizador
globoFit  = globoDataRaw.dropna(subset=["v"]).reset_index(drop=True)
pelotaFit = pelotaDataRaw.dropna(subset=["v"]).reset_index(drop=True)

def simulate(data, v0, m, k):
    """Corre RK4 sobre la malla de tiempo de 'data'; retorna arreglo de velocidades."""
    vSim = [v0]
    for i in range(len(data) - 1):
        h = data["t"].iloc[i+1] - data["t"].iloc[i]
        vSim.append(rungeKutta4(f, data["t"].iloc[i], vSim[-1], h, m, k))
    return np.array(vSim)

def fit_k(data, m, bounds):
    """Ajusta k minimizando la SSR entre la simulación RK4 y los datos experimentales."""
    v_exp = data["v"].values
    v0    = v_exp[0]

    def ssr(k):
        return np.sum((simulate(data, v0, m, k) - v_exp) ** 2)

    result = minimize_scalar(ssr, bounds=bounds, method="bounded")
    return result.x

# ajuste de k para cada objeto — bounds definidos por límites físicos de v_terminal
# globo:  v_term = mg/k → k en [mg/5, mg/0.1]  (v_term entre 0.1 y 5 m/s)
# pelota: v_term = mg/k → k en [mg/100, mg/5]  (v_term entre 5 y 100 m/s)
k_globo_opt  = fit_k(globoFit,  m=0.002, bounds=(0.002*9.81/5,   0.002*9.81/0.1))
k_pelota_opt = fit_k(pelotaFit, m=0.443, bounds=(0.443*9.81/100, 0.443*9.81/5))

v_term_globo  = (0.002 * 9.81) / k_globo_opt
v_term_pelota = (0.443 * 9.81) / k_pelota_opt

print(f"Globo  →  k = {k_globo_opt:.6f} N·s/m  |  v_term = {v_term_globo:.3f} m/s")
print(f"Pelota →  k = {k_pelota_opt:.6f} N·s/m  |  v_term = {v_term_pelota:.3f} m/s")
