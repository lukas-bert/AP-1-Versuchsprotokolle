import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const


d_lif = 201.4*10**-12


#überprüfung der braggbedingung
theta2, imp = np.genfromtxt("content/data/braggbedingung.txt", unpack = True)

plt.plot(theta2, imp, color = "firebrick", label = "Messung 1")
plt.tight_layout()
plt.grid()
#plt.xlabel(r'$\theta \mathbin{/} \unit{\degree}$')
#plt.ylabel(r'$\symup{Imp}\mathbin{/}\symup{s} $')
plt.legend()
#plt.savefig('build/plotbragg.pdf')
plt.show()
plt.close()

#Emissionsspektrum einer Cu-Röntgenröhre

theta2, imp = np.genfromtxt("content/data/emissionspektrum.txt", unpack = True)
#lambda1 = 2*d_lif*np.sin(theta2/2*np.pi/180)
plt.plot(theta2, imp, color = "firebrick", label = "Messung 2")
plt.plot(40.8, 1544.0, "x", markersize = 10 ,color = "cornflowerblue", label = "K_\alpha-Linie")
plt.plot(45.5, 5129.0, "x", markersize = 10 ,color = "cornflowerblue", label = "K_\beta-Linie")
plt.plot(theta2[46],imp[46], "o", markersize = 6, color = "cornflowerblue", label = "Bremsberg")
plt.plot(theta2[40:48],imp[40:48], "o", markersize = 6, color = "cornflowerblue", label = "Bremsberg")
plt.tight_layout()
plt.grid()
#plt.xlabel(r'$\theta \mathbin{/} \unit{\degree}$')
#plt.ylabel(r'$\symup{Imp}\mathbin{/}\symup{s} $')
plt.legend()
#plt.savefig('build/plotemission.pdf')
plt.show()
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

#experimentell
E_Kalphaexp = const.h*const.c/(2*d_lif*np.sin(40.8*np.pi/(2*180))*const.e*1000)
E_Kbetaexp = const.h*const.c/(2*d_lif*np.sin(45.4*np.pi/(2*180))*const.e*1000)

DeltaE_Kalphaexp = np.abs(const.h*const.c/(2*d_lif*np.sin(40.41154*np.pi/(2*180))*const.e*1000) - const.h*const.c/(2*d_lif*np.sin(41.308*np.pi/(2*180))*const.e*1000))
DeltaE_Kbetaexp = np.abs(const.h*const.c/(2*d_lif*np.sin(44.880769*np.pi/(2*180))*const.e*1000) - const.h*const.c/(2*d_lif*np.sin(45.83878469*np.pi/(2*180))*const.e*1000))
print(E_Kalphaexp, E_Kbetaexp, DeltaE_Kalphaexp,DeltaE_Kbetaexp)

#Auflösungsvermögen

A_Kalpha = E_Kalphaexp/DeltaE_Kalphaexp
A_Kbeta = E_Kbetaexp/DeltaE_Kbetaexp
print(A_Kalpha, A_Kbeta)


#HIER MUSS NOCH DIE GENAUIGKEIT DER ANGABE ANGEGEBEN WERDEN UND DER STATISTISCHE FEHLER BERECHNET WERDEN FALLS MAN SICH DAZU ENTSCHEIDET

#Abschirmkonstanten
E_abs = 8.988
sigma1 = 29 - np.sqrt(E_abs*1000/13.6 - (((7.297*10**-3)**2)*29**4)/4)
sigma2 = 29 - np.sqrt(E_Kalphaexp*1000/13.6 - (((7.297*10**-3)**2)*29**4)/4)
sigma3 = 29 - np.sqrt(E_Kbetaexp*1000/13.6 - (((7.297*10**-3)**2)*29**4)/4)
print(sigma1, sigma2, sigma3)