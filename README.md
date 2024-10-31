import numpy as np
import matplotlib.pyplot as plt

# Definir los parámetros
m_SB = 50           # kg
m_SA_inicial = 200  # kg
V = 1000            # L
Q_in = 30           # L/min
Q_out = 20          # L/min
dt = 10             # min
tiempo_total = 100  # tiempo en minutos para el cual queremos resolver la EDO

# Definir la ecuación diferencial
def dmSA_dt(m_SA):
    return (m_SB / V) * Q_in - (m_SA / V) * Q_out

# Implementar el método de Runge-Kutta de cuarto orden
def runge_kutta_4(f, y0, dt, tiempo_total):
    n_pasos = int(tiempo_total / dt)
    t = np.linspace(0, tiempo_total, n_pasos + 1)
    y = np.zeros(n_pasos + 1)
    y[0] = y0
    
    for i in range(n_pasos):
        k1 = f(y[i])
        k2 = f(y[i] + 0.5 * k1 * dt)
        k3 = f(y[i] + 0.5 * k2 * dt)
        k4 = f(y[i] + k3 * dt)
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
        
    return t, y

# Resolver la ecuación diferencial
t, mSA = runge_kutta_4(dmSA_dt, m_SA_inicial, dt, tiempo_total)

# Graficar la solución
plt.plot(t, mSA, label='m_SA(t)')
plt.xlabel('Tiempo (min)')
plt.ylabel('m_SA (kg)')
plt.title('Evolución de m_SA con el tiempo')
plt.grid()
plt.legend()
plt.show()

