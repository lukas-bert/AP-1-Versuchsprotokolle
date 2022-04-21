import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
#überprüfung der braggbedingung
theta2, imp = np.genfromtxt("content/data/braggbedingung.txt", unpack = True)

plt.plot(theta2, imp, color = "firebrick", label = "Messung 1")
plt.tight_layout()
plt.grid()
#plt.xlabel(r'$\theta \mathbin{/} \unit{\degree}$')
#plt.ylabel(r'$\symup{Imp}\mathbin{/}\symup{s} $')
plt.legend()
plt.savefig('build/plotbragg.pdf')
plt.close()

#Emissionsspektrum einer Cu-Röntgenröhre

theta2, imp = np.genfromtxt("content/data/emissionspektrum.txt", unpack = True)
plt.plot(theta2, imp, color = "firebrick", label = "Messung 2")
plt.plot(40.8, 1544.0, "x", markersize = 10 ,color = "cornflowerblue", label = "K_\alpha-Linie")
plt.plot(45.5, 5129.0, "x", markersize = 10 ,color = "cornflowerblue", label = "K_\beta-Linie")
plt.plot(theta2[46],imp[46], "o", markersize = 6, color = "cornflowerblue", label = "Bremsberg")
plt.tight_layout()
plt.grid()
#plt.xlabel(r'$\theta \mathbin{/} \unit{\degree}$')
#plt.ylabel(r'$\symup{Imp}\mathbin{/}\symup{s} $')
plt.legend()
plt.savefig('build/plotemission.pdf')
plt.close()

#GRENZWINKEL NOCH ZU BESTIMMEN ABER KA WIE DAS GEHEN SOLL BRATAN

#Detailspektrum

theta2, imp = np.genfromtxt("content/data/detailmessung.txt", unpack = True)
plt.plot(theta2, imp, color = "firebrick", label = "Messung Detail")
kalpha = np.linspace(40.41154, 41.308, 1000) #Intervall der Halbwertsbreite zu K_alpha
plt.plot(kalpha,760+kalpha*0)
kbeta = np.linspace(44.880769, 45.83878469, 1000) #Intervall der Halbwertsbreite zu K_beta
plt.plot(kbeta,2641.5+kbeta*0)
plt.show()
plt.close()

##Berechnung des AUflösungsvermögens
d_lif = 201.4*10**-12
#experimentell
E_Kalphaexp = const.h*const.c/(2*d_lif*np.sin(40.8/2)*const.e*1000)
E_Kbetaexp = const.h*const.c/(2*d_lif*np.sin(45.4/2)*const.e*1000)

DeltaE_Kalphaexp = np.abs(const.h*const.c/(2*d_lif*np.sin(40.41154/2)*const.e*1000) - const.h*const.c/(2*d_lif*np.sin(41.308/2)*const.e*1000))
DeltaE_Kbetaexp = np.abs(const.h*const.c/(2*d_lif*np.sin(44.880769/2)*const.e*1000) - const.h*const.c/(2*d_lif*np.sin(45.83878469/2)*const.e*1000))
print(E_Kalphaexp, E_Kbetaexp, DeltaE_Kalphaexp,DeltaE_Kbetaexp)

#Auflösungsvermögen

A_Kalpha = E_Kalphaexp/DeltaE_Kalphaexp
A_Kbeta = E_Kbetaexp/DeltaE_Kbetaexp
print(A_Kalpha, A_Kbeta)
#DIE WERTE SIND TOTTAL KAKKE ABER ICH FINDE DEN FEHLER NICHT

#HIER MUSS NOCH DIE GENAUIGKEIT DER ANGABE ANGEGEBEN WERDEN UND DER STATISTISCHE FEHLER BERECHNET WERDEN FALLS MAN SICH DAZU ENTSCHEIDET