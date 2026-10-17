######################################################################
#        AJUSTE DE k CON LA SOLUCIÓN ANALÍTICA DE  m v' = m g - k v
#        (Código asistido por la herramienta Claude Code)
#
#  Resolviendo la EDO por variables separables con v(t0) = v0:
#
#        v(t) = mg/k + (v0 - mg/k) * exp(-k (t - t0) / m)
#
#  En lugar de integrar numéricamente (Euler/RK4, ver rk4Sim.py y
#  runge_kutta_app.py), se ajusta k directamente sobre esta expresión
#  cerrada con mínimos cuadrados no lineales (scipy.optimize.curve_fit).
######################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

g = 9.81

# extracción de la data del globo y de la pelota (mismo formato que rk4Sim.py)
globoDataRaw = pd.read_csv("Tracker_globo_Raw_Data.txt", sep=";", skiprows=2, names=["t", "y", "v", "a"], usecols=[0,1,2,3])
pelotaDataRaw = pd.read_csv("Tracker_pelota_Raw_Data.txt", sep=";", skiprows=2, names=["t", "y", "v", "a"], usecols=[0,1,2,3])

# se quitan el primer y último dato (Tracker no calcula v en los extremos)
globoData = globoDataRaw.iloc[1:-1].dropna(subset=["v"]).reset_index(drop=True)
pelotaData = pelotaDataRaw.iloc[1:-1].dropna(subset=["v"]).reset_index(drop=True)


def vAnalitica(t, k, m, t0, v0):
    """Solución particular de m v' = m g - k v con v(t0) = v0."""
    vTerm = m * g / k
    return vTerm + (v0 - vTerm) * np.exp(-k * (t - t0) / m)


def ajustarK(data, m, kInicial):
    """Ajusta k por mínimos cuadrados; retorna k, su incertidumbre, RMSE y R²."""
    t = data["t"].values
    v = data["v"].values
    t0, v0 = t[0], v[0]

    modelo = lambda t, k: vAnalitica(t, k, m, t0, v0)
    (k,), cov = curve_fit(modelo, t, v, p0=[kInicial], bounds=(1e-8, np.inf))

    residuos = v - modelo(t, k)
    rmse = np.sqrt(np.mean(residuos ** 2))
    r2 = 1 - np.sum(residuos ** 2) / np.sum((v - v.mean()) ** 2)
    return k, np.sqrt(cov[0, 0]), rmse, r2


objetos = {
    "Globo":  {"data": globoData,  "m": 0.002, "kInicial": 0.01},
    "Pelota": {"data": pelotaData, "m": 0.443, "kInicial": 0.1},
}

for nombre, obj in objetos.items():
    k, dk, rmse, r2 = ajustarK(obj["data"], obj["m"], obj["kInicial"])
    obj.update(k=k, dk=dk, vTerm=obj["m"] * g / k)
    print(f"{nombre:6s} →  k = {k:.6f} ± {dk:.6f} N·s/m  |  v_term = {obj['vTerm']:.3f} m/s"
          f"  |  RMSE = {rmse:.3f} m/s  |  R² = {r2:.4f}")

######################################################################
#                              GRÁFICA
######################################################################

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, (nombre, obj) in zip(axes, objetos.items()):
    data, m, k = obj["data"], obj["m"], obj["k"]
    t0, v0 = data["t"].iloc[0], data["v"].iloc[0]
    tFino = np.linspace(t0, data["t"].iloc[-1], 500)

    ax.scatter(data["t"], data["v"], s=14, color="steelblue", label="Datos experimentales (Tracker)", zorder=3)
    ax.plot(tFino, vAnalitica(tFino, k, m, t0, v0), color="darkorange", lw=2.2,
            label=f"Solución analítica ajustada  (k = {k:.4f} N·s/m)", zorder=4)
    ax.plot(tFino, v0 + g * (tFino - t0), "--", color="seagreen", lw=1.5, label="Caída libre  (sin arrastre)")
    ax.axhline(obj["vTerm"], ls=":", color="gray", lw=1, label=f"$v_{{term}}$ = {obj['vTerm']:.2f} m/s")

    ax.set_title(nombre)
    ax.set_xlabel("$t$ (s)")
    ax.set_ylabel("$v$ (m/s)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

fig.suptitle(r"Ajuste por solución analítica — $v(t) = \frac{mg}{k} + \left(v_0 - \frac{mg}{k}\right)e^{-k(t-t_0)/m}$")
plt.tight_layout()
plt.savefig("Media/v_vs_t_analitico.png", dpi=150, bbox_inches="tight")
plt.show()
