import pandas as pd

# extracción de la data del globo
globoDataRaw = pd.read_csv("Tracker_globo_Raw_Data.txt", sep=";", skiprows=2, names=["t", "y", "v", "a"], usecols=[0,1,2,3])

# quiero quitar los primeros y ultimos datos porque son basicamente basura y no me sirven mucho
globoDataRaw.drop(index=globoDataRaw.index[0], axis=0, inplace=True)
globoDataRaw.drop(index=globoDataRaw.index[-1], axis=0, inplace=True)

print(globoDataRaw.to_string())

# definicion de la EDO para caida libre v'=g-(k/m)v
def f(t, v, m, k):
    return 9.81-(k/m)*v

# funcion para el modelo de
def rungeKutta4(f,t,v,h,m,k):
    k1 = f(t,v,m,k)
    k2 = f(t+h/2, v+h*k1/2,m,k)
    k3 = f(t+h/2, v+h*k2/2,m,k)
    k4 = f(t+h, v+h*k3,m,k)
    return v+(h/6) * (k1+2*k2+2*k3+k4)

vSim = [0.051429]
# simulacion para la trayectoria completa
for i in range(len(globoDataRaw)-1):
    h = globoDataRaw["t"].iloc[i+1] - globoDataRaw["t"].iloc[i]
    # aca voy a tomar algunos supuestos que despues se van a optimizar:
        # un globo de latex estándar tiene entre 10 a 19 gramos de masa, el promedio de eso seria 14.5g.
        # sabiendo que a velocidad terminal, v'=0, podemos despejar para hacer una suposicion inicial de 'k'
        # k = mg/v_t, pero como no se llega a la velocidad terminal en todo el trayecto agarré el ultimo dato que pude, resultando en k = 0.0805.
    vNext = rungeKutta4(f, globoDataRaw["t"].iloc[i], vSim[-1], h, 0.0145, 0.0805)
    vSim.append(vNext)

for j in range(len(vSim)-1):
    print(f"Velocity no.{j}: {vSim[j]}")
