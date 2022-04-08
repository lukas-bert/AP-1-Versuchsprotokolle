import matplotlib.pyplot as plt
import numpy as np

import uncertainties.unumpy as unp
import uncertainties as unc
from uncertainties import ufloat
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as devs

# Konstanten
c_acryl = 2730                  # m/s Ausbreitungsgeschwindigkeit in Acryl
c_destwasser = 1483             # m/s Ausbreitungsgeschwindigkeit in destilliertem Wasser
height = 80.55                  # Höhe des Blockes in mm

# Reale Abmessungen
d, tiefe = np.genfromtxt("content/Abmessungen.txt", unpack = True)  # Durchmesser in mm, vertikale Tiefe der Bohrungen in mm
real_o = height - tiefe    # Umrechnen in Höhe auf y-Skala
pos = [1, 2, 3, 4, 5, 6, 7, 8, 8.2, 0.5, 0.6]                  # Array zur Darstellung der Positionen der Löcher

# Bestimmung der Dicke der Kontaktmittel- / Anpassungsschicht aus Messreihe 1
t_1 = np.genfromtxt("content/Messung1.txt", unpack = True)          # Laufzeiten aus Messreihe 1 in µs

real_temp = np.zeros(7)         # zugehörige echte Daten der Tiefe (Löcher 1-6 + Loch 9)
for i in range(6):
    real_temp[i] = tiefe[i]
real_temp[6] = tiefe[8]    

t_real = 2 * real_temp/c_acryl * 10**3  # Faktor 2, wgn hin- und Rückweg,  10**3 --> µs

time_diff = abs(t_real - t_1)

time_diff_mean = np.mean(time_diff)
time_diff_std = np.std(time_diff)
time_diff_err = time_diff_std/3               # Fehler des Mittelwerts (Messumfang = 9 --> sqrt(N=9)=3)
u_time_diff = ufloat(time_diff_mean, time_diff_err)

# Dicke der Kontaktmittelschicht:
kontakt = c_destwasser * u_time_diff/2 *10**(-3)

print("Dicke der Kontaktmittelschicht nach Messung 1: ", kontakt, time_diff_mean)

# Plot der realen Positionen
o, u = np.genfromtxt("content/Messung2.txt", unpack = True)     # Messwerte der Tiefenmessung (oben, unten)
o = o - noms(kontakt)                                           # Subtraktion der Kontaktmittelschicht
u = u - noms(kontakt)

d_mess = height - o - u
print(d_mess)

o = height - o

for i in range(11):     # for Schleife für variable Marker Größe
    plt.plot(pos[i], real_o[i] - 1/2*d[i], "o", markersize = 4*d[i], color = "cornflowerblue")
    plt.plot(pos[i], real_o[i], "_", markersize = 15, color = "mediumblue")
    plt.plot(pos[i], real_o[i] - d[i], "_", markersize = 15, color = "mediumblue")

plt.plot(pos[10], real_o[10], "_", markersize = 15, color = "mediumblue", label = "Reale Messwerte")

# US-Scan Plot                    (Mit Bereinigung der Abweichungen durch Kontaktmittelschicht)

plt.plot(pos, o, "_", markersize = 15 ,color = "firebrick", label = "Aus US berechnete Positionen")
plt.plot(pos, u, "_", markersize = 15 ,color = "firebrick")

for i in range(11):     # for Schleife für variable Marker Größe (US-Daten)
    plt.plot(pos[i], o[i] - 1/2*d_mess[i], "o", markersize = 4*d[i], color = "firebrick", alpha=0.3)

plt.legend()

plt.show()
#plt.savefig('build/plot.pdf')
plt.close()

##Differenz der Signallaufzeit zwischen Theorie- und Messwert(o=oben,u=unten,t=theorie,n=nummer_der_störstelle):
#time_otn = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
#time_utn = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
#time_odiffn = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
#time_udiffn = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
#for n in range(11):
#    time_otn[n] = 2*(t[n]/c_acryl)*10**3 #microsekunden
#    time_utn[n] = 2*(t_u[n]/c_acryl)*10**3 #microsekunden
#    time_odiffn[n] = np.abs(time_otn[n] - time_o[n])
#    time_udiffn[n] = np.abs(time_utn[n] - time_u[n])
##timediffo_mean = np.mean(time_odiffn)
#timediffu_mean = np.mean(time_udiffn)
##timediffo_std = np.std(time_odiffn) #muss man ma schauen wie man das macht
#timediffu_std = np.std(time_udiffn) 
#
##testeroni:
#timediffo_mean = 0
#for s in range(7):
#    timediffo_mean = timediffo_mean + time_odiffn[s]
#for s in range(3):
#    timediffo_mean = timediffo_mean + time_odiffn[s+8]
#timediffo_mean = timediffo_mean/10
##Berechnung der Dicke der Anpassungsschicht:
#bo_destwasser = c_destwasser*timediffo_mean*10**-3 #in mm
#bu_destwasser = c_destwasser*timediffu_mean*10**-3 #in mm
#print(bo_destwasser,bu_destwasser)

time_o, time_u = np.genfromtxt("content/Messung3.txt", unpack = True)
time_o = time_o - time_diff_mean
time_u = time_u - time_diff_mean

o = height - time_o*c_acryl/2 * 10**(-3) #- noms(kontakt)
u =  time_u*c_acryl/2 * 10**(-3) #- noms(kontakt)

for i in range(11):     # for Schleife für variable Marker Größe
    plt.plot(pos[i], real_o[i] - 1/2*d[i], "o", markersize = 4*d[i], color = "cornflowerblue")
    plt.plot(pos[i], real_o[i], "_", markersize = 15, color = "mediumblue")
    plt.plot(pos[i], real_o[i] - d[i], "_", markersize = 15, color = "mediumblue")

# US-Scan Plot                    (Mit Bereinigung der Abweichungen durch Kontaktmittelschicht)

plt.plot(pos, o, "_", markersize = 15 ,color = "firebrick")
plt.plot(pos, u, "_", markersize = 15 ,color = "firebrick")

plt.close()
