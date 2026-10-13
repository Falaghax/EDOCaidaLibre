import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
print(globoDataRaw.to_string())
print(pelotaDataRaw.to_string())

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

def simGlobo():
    # 0.051429 es la velocidad inicial que aparece en nuestros datos
    vSim = [0.051429]
    # simulacion para la trayectoria completa
    for i in range(len(globoDataRaw)-1):
        h = globoDataRaw["t"].iloc[i+1] - globoDataRaw["t"].iloc[i]
        # aca voy a tomar algunos supuestos que despues se van a optimizar usando IA:
            # Iniciamos sabiendo que el globo tiene una masa de 2 gramos cuando está inflado.
            # Sabiendo que a velocidad terminal, v'=0, podemos despejar para hacer una suposicion inicial de 'k'
            # k = mg/v_t, agarré el ultimo dato de velocidad que
            # aparece en la data cruda, resultando en k = 0.0111 N*s/m.
        vNext = rungeKutta4(f, globoDataRaw["t"].iloc[i], vSim[-1], h, 0.002, 0.0111)
        vSim.append(vNext)
    print("################################################################################")
    print("PREDICCIONES DE VELOCIDAD DEL GLOBO:")
    for j in range(len(vSim)-1):
        print(f"Velocity no.{j}: {vSim[j]} m/s")
    print("################################################################################")

def simPelota():
    # 0.051429 es la velocidad inicial que aparece en nuestros datos
    vSimPelota = [0.460433]
    # simulacion para la trayectoria completa
    for x in range(len(pelotaDataRaw)-1):
        h = pelotaDataRaw["t"].iloc[x+1] - pelotaDataRaw["t"].iloc[x]
        #usé el mismo calculo para los datos de acá
        vNext = rungeKutta4(f, pelotaDataRaw["t"].iloc[x], vSimPelota[-1], h, 0.443, 0.2806)
        vSimPelota.append(vNext)
    print("################################################################################")
    print("PREDICCIONES DE VELOCIDAD DE LA PELOTA:")
    for y in range(len(vSimPelota)-1):
        print(f"Velocity no.{y}: {vSimPelota[y]} m/s")
    print("################################################################################")

# llamada a las funciones respectivas
simGlobo()
simPelota()

######################################################################
#                 AJUSTE AUTOMÁTICO DE k Y GRÁFICAS
#                 (A partir de aqui, el código, no es de mi autoría,
#                 fue asistido por parte de la herramienta Claude Code)
######################################################################

# Datos limpios (sin filas con v=NaN) para el optimizador y las gráficas
globoFit  = globoDataRaw.dropna(subset=["v"]).reset_index(drop=True)
pelotaFit = pelotaDataRaw.dropna(subset=["v"]).reset_index(drop=True)

def simulate(data, v0, m, k):
    """Corre RK4 sobre la malla de tiempo de 'data'; retorna arreglo de velocidades."""
    vSim = [v0]
    for i in range(len(data) - 1):
        h = data["t"].iloc[i+1] - data["t"].iloc[i]
        vSim.append(rungeKutta4(f, data["t"].iloc[i], vSim[-1], h, m, k))
    return np.array(vSim)

def free_fall(data, v0):
    """Caída libre analítica sin arrastre: v = v₀ + g·(t − t₀)"""
    return v0 + 9.81 * (data["t"].values - data["t"].iloc[0])

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

# simulaciones con k optimizado y caída libre
v_rk4_globo  = simulate(globoFit,  globoFit["v"].iloc[0],  0.002, k_globo_opt)
v_ff_globo   = free_fall(globoFit,  globoFit["v"].iloc[0])

v_rk4_pelota = simulate(pelotaFit, pelotaFit["v"].iloc[0], 0.443, k_pelota_opt)
v_ff_pelota  = free_fall(pelotaFit, pelotaFit["v"].iloc[0])

# gráficas v vs t
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

datasets = [
    (axes[0], globoFit,  v_rk4_globo,  v_ff_globo,  "Globo",  k_globo_opt,  v_term_globo),
    (axes[1], pelotaFit, v_rk4_pelota, v_ff_pelota, "Pelota", k_pelota_opt, v_term_pelota),
]

for ax, data, v_rk4, v_ff, label, k_opt, v_term in datasets:
    t     = data["t"].values
    v_exp = data["v"].values

    ax.scatter(t, v_exp, s=18, color="steelblue", zorder=3,
               label="Datos experimentales (Tracker)")
    ax.plot(t, v_rk4, color="tomato",   lw=2,
            label=f"RK4 con arrastre  (k = {k_opt:.4f} N·s/m)")
    ax.plot(t, v_ff,  color="seagreen", lw=1.5, ls="--",
            label="Caída libre  (sin arrastre)")
    ax.axhline(v_term, color="gray", lw=1, ls=":",
               label=f"$v_{{term}}$ = {v_term:.2f} m/s")

    ax.set_title(label)
    ax.set_xlabel("$t$ (s)")
    ax.set_ylabel("$v$ (m/s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle("Resistencia dinámica de fuerzas viscosas — $mv' = mg - kv$", fontsize=13)
plt.tight_layout()
plt.savefig("v_vs_t.png", dpi=150, bbox_inches="tight")
plt.show()
