import matplotlib.pyplot as plt
import numpy as np

c_acryl = 2730 #m/s ausbreitungsgeschwindigkeit in acryl
c_destwasser = 1483 #m/s ausbreitungsgeschwindigkeit in destilliertem Wasser
height = 80.55                                                  # Höhe des Blockes in mm
d, t = np.genfromtxt("content/Abmessungen.txt", unpack = True)  # Durchmesser, vertikale Tiefe der Bohrungen
o, u = np.genfromtxt("content/Messung2.txt", unpack = True)     # Messwerte der Tiefenmessung (oben, unten)
time_o, time_u = np.genfromtxt("content/Messung3.txt", unpack = True)     # Messwerte der SIgnallaufdauer der Tiefenmessung (oben, unten)

o = height - o #könnte gleichen fehler wie unter diesem aufweisen da bei beiden der durchmesser nicht betrachtet wurde!!!!!!!!!!!!!!!!
t_u = height - t - d #habe hier ein t_u ersetzt da dieser wert eig der von unten seien sollte müssen dann aber nochmal nachsehen wegen dem plot den ich nicht ändere
time_oo = 2*(height/c_acryl)*10**3 - time_o 

pos = [1, 2, 3, 4, 5, 6, 7, 8, 8.2, 0.5, 0.6]                  # Array zur Darstellung der Positionen der Löcher

#Differenz der Signallaufzeit zwischen Theorie- und Messwert(o=oben,u=unten,t=theorie,n=nummer_der_störstelle):
time_otn = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
time_utn = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
time_odiffn = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
time_udiffn = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
for n in range(11):
    time_otn[n] = 2*((t[n]*10**(-3))/c_acryl)*10**6 #microsekunden
    time_utn[n] = 2*((t_u[n]*10**(-3))/c_acryl)*10**6 #microsekunden
    time_odiffn[n] = np.abs(time_otn[n] - time_o[n])
    time_udiffn[n] = np.abs(time_utn[n] - time_u[n])
timediffo_mean = np.mean(time_odiffn)
timediffu_mean = np.mean(time_udiffn)
timediffo_std = np.std(time_odiffn)
timediffu_std = np.std(time_udiffn)

#Berechnung der Dicke der Anpassungsschicht:
bo_destwasser = c_destwasser*timediffo_mean*10**-3 #in mm
bu_destwasser = c_destwasser*timediffu_mean*10**-3 #in mm
print(bo_destwasser,bu_destwasser)

# Plot der realen Positionen (variable Marker Größe)

for i in range(11):
    plt.plot(pos[i], t_u[i] + 1/2*d[i], "o", markersize = 4*d[i], color = "cornflowerblue")
    plt.plot(pos[i], t_u[i], "_", markersize = 10, color = "mediumblue")
    plt.plot(pos[i], t_u[i] + d[i], "_", markersize = 10, color = "mediumblue")

# US-Scan Plot                    !!!!!(Noch ohne Bereinigung von Kontaktmittelschicht)

plt.plot(pos, o, "_", markersize = 10 ,color = "firebrick")
plt.plot(pos, u, "_", markersize = 10 ,color = "firebrick")

plt.show()
plt.close()

for i in range(11):
    plt.plot(pos[i], 2*((t_u[i] + 1/2*d[i])/c_acryl)*10**3, "o", markersize = 4*d[i], color = "cornflowerblue")
    plt.plot(pos[i], 2*(t_u[i]/c_acryl)*10**3, "_", markersize = 10, color = "mediumblue")
    plt.plot(pos[i], 2*((t_u[i] + d[i])/c_acryl)*10**3, "_", markersize = 10, color = "mediumblue")

plt.plot(pos, time_oo, "_", markersize = 10 ,color = "firebrick")
plt.plot(pos, time_u, "_", markersize = 10 ,color = "firebrick")
plt.show()
plt.close()
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
