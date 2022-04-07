import matplotlib.pyplot as plt
import numpy as np

height = 80.55                                                  # Höhe des Blockes in mm
d, t = np.genfromtxt("content/Abmessungen.txt", unpack = True)  # Durchmesser, vertikale Tiefe der Bohrungen
o, u = np.genfromtxt("content/Messung2.txt", unpack = True)     # Messwerte der Tiefenmessung (oben, unten)

o = height - o
t = height - t

pos = [1, 2, 3, 4, 5, 6, 7, 8, 7.5, 0.5, 0.6]                  # Array zur Darstellung der Positionen der Löcher


# Plot der realen Positionen (variable Marker Größe)

for i in range(11):
    plt.plot(pos[i], t[i] + 1/2*d[i], "o", markersize = 4*d[i], color = "cornflowerblue")
    plt.plot(pos[i], t[i], "_", markersize = 10, color = "mediumblue")
    plt.plot(pos[i], t[i] + d[i], "_", markersize = 10, color = "mediumblue")

# US-Scan Plot                    !!!!!(Noch ohne Bereinigung von Kontaktmittelschicht)

plt.plot(pos, o, "_", markersize = 10 ,color = "firebrick")
plt.plot(pos, u, "_", markersize = 10 ,color = "firebrick")

plt.show()

#x = np.linspace(0, 10, 1000)
#y = x ** np.sin(x)
#
#plt.subplot(1, 2, 1)
#plt.plot(x, y, label='Kurve')
#plt.xlabel(r'$\alpha \mathbin{/} \unit{\ohm}$')
#plt.ylabel(r'$y \mathbin{/} \unit{\micro\joule}$')
#plt.legend(loc='best')
#
#plt.subplot(1, 2, 2)
#plt.plot(x, y, label='Kurve')
#plt.xlabel(r'$\alpha \mathbin{/} \unit{\ohm}$')
#plt.ylabel(r'$y \mathbin{/} \unit{\micro\joule}$')
#plt.legend(loc='best')
#
## in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
#plt.savefig('build/plot.pdf')
