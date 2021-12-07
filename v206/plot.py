import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import uncertainties.unumpy as unp
from uncertainties import ufloat
import scipy.constants as const

t, T_2, T_1 , p_a, p_b , P = np.genfromtxt("content/data.txt", unpack = True)

p_a = p_a*10**5
p_b = p_b*10**5
t = 60*t
T_2 = T_2 + 273.15
T_1 = T_1 + 273.15  # Umrechnen in SI

def f1(t, A, B, C):
    return A*t**2 + B*t + C

def f1_dt(t, A, B):
    return 2*A*t + B

#def f2(t, A, B, a):
#    return A/(1*B*t**a)
#
#def f3(t, A, B, C, a):
#    return ((A*(t**a))/(1*B*(t**a))) + C

# Aufgabe a),b)

params1, pcov1 = op.curve_fit(f1, t, T_2)
params2, pcov2 = op.curve_fit(f1, t, T_1)

x = np.linspace(0, 1300, 1000)

plt.plot(x, f1(x, params1[0], params1[1], params1[2]), 'c', label = "Fit zum kalten Gefäß")
plt.plot(x, f1(x, params2[0], params2[1], params2[2]), 'tab:orange', label = "Fit zum warmen Gefäß")
plt.errorbar(t, T_2, xerr = 0, yerr = 0.1, fmt = 'bx', label = 'Messwerte kaltes Gefäß')
plt.errorbar(t, T_1, xerr = 0, yerr = 0.1, fmt = 'rx', label = 'Messwerte warmes Gefäß')
plt.xlabel("t in s")
plt.ylabel("T in K")
plt.xlim(0, 1250)
plt.legend(loc='best')
plt.savefig('build/plot.pdf')
plt.close()

# Aufgabe c)

dt = np.array([3, 8, 13, 18])
dT2 = f1_dt(dt, params1[0], params1[1])
dT1 = f1_dt(dt, params2[0], params2[1])

print(dt, dT1, dT2)

# c_wasser = 4.1818 kJ/(kg*K)
c_w = 4181.8
dQ1 = (3*c_w + 750)*dT1

v1 = dQ1[0]/P[2]
v2 = dQ1[1]/P[7]
v3 = dQ1[2]/P[12]
v4 = dQ1[3]/P[17]

v1t = T_1[2]/(T_1[2]-T_2[2])
v2t = T_1[7]/(T_1[7]-T_2[7])
v3t = T_1[12]/(T_1[12]-T_2[12])
v4t = T_1[17]/(T_1[17]-T_2[17])

print("Experiment:  ", v1, v2, v3, v4)
print("Theorie      ", v1t, v2t, v3t, v4t)

# in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
#plt.savefig('build/plot.pdf')