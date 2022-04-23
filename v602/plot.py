import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import scipy.optimize as op

d_lif = 201.4*10**-12 # in m
R = 13.6 # in eV
alpha = 7.297*10**-3

#überprüfung der braggbedingung
theta2, imp = np.genfromtxt("content/data/braggbedingung.txt", unpack = True)

#abweichung vom Theoriewinkel
abw1 = np.abs(27.3 - 28)/28
print("Abweichung des maximums der Braggbedingung: ", abw1)

plt.plot(theta2, imp, color = "firebrick", label = "Messung 1")
plt.plot(27.3, 221.0, "o", markersize = 10 ,color = "cornflowerblue", label = "Maximum")
plt.grid()
plt.xlabel(r'$2\theta \mathbin{/} \unit{\degree}$')
plt.ylabel(r'$\symup{Imp}\mathbin{/}\symup{s} $')
plt.legend()
plt.tight_layout()
plt.savefig('build/plotbragg.pdf')
#plt.show()
plt.close()

#Emissionsspektrum einer Cu-Röntgenröhre

theta2, imp = np.genfromtxt("content/data/emissionspektrum.txt", unpack = True)
#lambda1 = 2*d_lif*np.sin(theta2/2*np.pi/180)
plt.plot(theta2, imp, color = "firebrick", label = "Messung 2")
plt.plot(40.8, 1544.0, "x", markersize = 10 ,color = "cornflowerblue", label = r"$K_\alpha\text{-Linie}$")
plt.plot(45.5, 5129.0, "x", markersize = 10 ,color = "cornflowerblue", label = r"$K_\beta\text{-Linie}$")
plt.plot(theta2[46],imp[46], "o", markersize = 6, color = "cornflowerblue", label = "Bremsberg")
#plt.plot(theta2[40:48],imp[40:48], "o", markersize = 6, color = "cornflowerblue", label = "Bremsberg")
plt.grid()
plt.xlabel(r'$2\theta \mathbin{/} \unit{\degree}$')
plt.ylabel(r'$\symup{Imp}\mathbin{/}\symup{s} $')
plt.legend()
plt.tight_layout()
plt.savefig('build/plotemission.pdf')
#plt.show()
plt.close()

#GRENZWINKEL NOCH ZU BESTIMMEN ABER KA WIE DAS GEHEN SOLL BRATAN

#Detailspektrum

theta2, imp = np.genfromtxt("content/data/detailmessung.txt", unpack = True)
plt.plot(theta2, imp, color = "firebrick", label = "Messung Detail")
plt.xlabel(r'$2\theta \mathbin{/} \unit{\degree}$')
plt.ylabel(r'$\symup{Imp}\mathbin{/}\symup{s} $')
kalpha = np.linspace(40.41154, 41.308, 1000) #Intervall der Halbwertsbreite zu K_alpha
plt.plot(kalpha,760+kalpha*0, label = r"$\text{Halbwertsbreite des } K_{\beta}\text{-Peaks}$")
kbeta = np.linspace(44.880769, 45.83878469, 1000) #Intervall der Halbwertsbreite zu K_beta
plt.plot(kbeta,2641.5+kbeta*0, label = r"$\text{Halbwertsbreite des } K_{\alpha}\text{-Peaks}$")
plt.grid()
plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig('build/detailspektrum.pdf')
plt.close()

##Berechnung des AUflösungsvermögens

#experimentell
E_Kbetaexp = const.h*const.c/(2*d_lif*np.sin(40.8*np.pi/(2*180))*const.e*1000)
E_Kalphaexp = const.h*const.c/(2*d_lif*np.sin(45.4*np.pi/(2*180))*const.e*1000)

DeltaE_Kbetaexp = np.abs(const.h*const.c/(2*d_lif*np.sin(40.41154*np.pi/(2*180))*const.e*1000) - const.h*const.c/(2*d_lif*np.sin(41.308*np.pi/(2*180))*const.e*1000))
DeltaE_Kalphaexp = np.abs(const.h*const.c/(2*d_lif*np.sin(44.880769*np.pi/(2*180))*const.e*1000) - const.h*const.c/(2*d_lif*np.sin(45.83878469*np.pi/(2*180))*const.e*1000))
print(E_Kalphaexp, E_Kbetaexp, DeltaE_Kalphaexp,DeltaE_Kbetaexp)

#Auflösungsvermögen

A_Kalpha = E_Kalphaexp/DeltaE_Kalphaexp
A_Kbeta = E_Kbetaexp/DeltaE_Kbetaexp
print(A_Kalpha, A_Kbeta)

#abschirmkonstanten
sigma1 = 29 - np.sqrt(8.988*1000/R)
print("sigma_1 von Kupfer: ", sigma1)
sigma2 = 29 - np.sqrt(4*(29 - sigma1)**2 - 4*(E_Kalphaexp*1000/R))
print("sigma_2 von Kupfer: ", sigma2)
sigma3 = 29 - np.sqrt(9*(29 - sigma1)**2 - 9*(E_Kbetaexp*1000/R))
print("sigma_3 von Kupfer: ", sigma3)


#theorie
sigma1 = 29 - np.sqrt(8.988*1000/R)
print("sigma_1t von Kupfer: ", sigma1)
sigma2 = 29 - np.sqrt(4*(29 - sigma1)**2 - 4*(8*1000/R))
print("sigma_2t von Kupfer: ", sigma2)
sigma3 = 29 - np.sqrt(9*(29 - sigma1)**2 - 9*(8.95*1000/R))
print("sigma_3t von Kupfer: ", sigma3)



#Abschirmkonstanten
E_abs = 8.988
sigma1 = 29 - np.sqrt(E_abs*1000/13.6 - (((7.297*10**-3)**2)*29**4)/4)
sigma2 = 29 - np.sqrt(E_Kalphaexp*1000/13.6 - (((7.297*10**-3)**2)*29**4)/4)
sigma3 = 29 - np.sqrt(E_Kbetaexp*1000/13.6 - (((7.297*10**-3)**2)*29**4)/4)
print(sigma1, sigma2, sigma3)


#absorptionsspektrum

def sigmak(Z, E):
    return Z - np.sqrt((E*1000/R) - (((alpha**2)*(Z**4))/4))


#Brom
Z = 35
theta2, imp = np.genfromtxt("content/data/Br35.txt", unpack = True)
plt.subplot(3, 2, 3)
plt.plot(theta2, imp, color = "firebrick", label = "Br35")
imp_mittel = imp[0] + (imp[-1] - imp[0])/2 
x = np.linspace(theta2[0],theta2[-1],1000)
plt.plot(26.8, imp_mittel, "x", color = "cornflowerblue" ,label = r"$E_{\text{abs}}$")

plt.grid()
plt.xlabel(r'$2\theta \mathbin{/} \unit{\degree}$')
plt.ylabel(r'$\symup{Imp}\mathbin{/}\symup{s} $')
plt.legend()
plt.tight_layout()
#plt.savefig('build/Br35.pdf')
#plt.show()
#plt.close()

E_absorbbr = const.h*const.c/(2*d_lif*np.sin(26.8*np.pi/(2*180))*const.e*1000)
sigma_k = sigmak(Z, E_absorbbr)
print("sigma_k für Br: ", sigma_k)
print("E_abs für Br : ", E_absorbbr)

#Gallium

Z = 31

theta2, imp = np.genfromtxt("content/data/Ga31.txt", unpack = True)
plt.subplot(3, 2, 2)
plt.plot(theta2, imp, color = "firebrick", label = "Ga31")
imp_mittel = imp[0] + (imp[-1] - imp[0])/2 
x = np.linspace(theta2[0],theta2[-1],1000)
plt.plot(35.0133, imp_mittel, "x", color = "cornflowerblue", label = r"$E_{\text{abs}}$")

plt.grid()
plt.xlabel(r'$2\theta \mathbin{/} \unit{\degree}$')
plt.ylabel(r'$\symup{Imp}\mathbin{/}\symup{s} $')
plt.legend()
plt.tight_layout()
#plt.savefig('build/plotbragg.pdf')
#plt.show()
#plt.close()

E_absorbga = const.h*const.c/(2*d_lif*np.sin(35.0133*np.pi/(2*180))*const.e*1000)
sigma_k = sigmak(Z, E_absorbga)
print("sigma_k für Ga: ", sigma_k)
print("E_abs für Ga : ", E_absorbga)


#Strontium 
Z = 38

theta2, imp = np.genfromtxt("content/data/Sr38.txt", unpack = True)
plt.subplot(3, 2, 4)
plt.plot(theta2, imp, color = "firebrick", label = "Zr38")
imp_mittel = imp[0] + (imp[-1] - imp[0])/2 
x = np.linspace(theta2[0],theta2[-1],1000)
plt.plot(22.7214, imp_mittel, "x", color = "cornflowerblue", label = r"$E_{\text{abs}}$")

plt.grid()
plt.xlabel(r'$2\theta \mathbin{/} \unit{\degree}$')
plt.ylabel(r'$\symup{Imp}\mathbin{/}\symup{s} $')
plt.legend()
plt.tight_layout()
#plt.savefig('build/plotbragg.pdf')
#plt.show()
#plt.close()

E_absorbsr = const.h*const.c/(2*d_lif*np.sin(22.7214*np.pi/(2*180))*const.e*1000)
sigma_k = sigmak(Z, E_absorbsr)
print("sigma_k für Sr: ", sigma_k)
print("E_abs für Sr : ", E_absorbsr)


#Zn
Z = 30
theta2, imp = np.genfromtxt("content/data/Zn30.txt", unpack = True)
plt.subplot(3, 2, 1)
plt.plot(theta2, imp, color = "firebrick", label = "Zn30")
imp_mittel = imp[0] + (imp[-1] - imp[0])/2 
x = np.linspace(37,38,1000)
plt.plot(37.56, imp_mittel, "x", color = "cornflowerblue", label = r"$E_{\text{abs}}$")

plt.grid()
plt.xlabel(r'$2\theta \mathbin{/} \unit{\degree}$')
plt.ylabel(r'$\symup{Imp}\mathbin{/}\symup{s} $')
plt.legend()
plt.tight_layout()
#plt.savefig('build/plotbragg.pdf')
#plt.show()
#plt.close()



E_absorbzn = const.h*const.c/(2*d_lif*np.sin(37.56*np.pi/(2*180))*const.e*1000)
sigma_k = sigmak(Z, E_absorbzn)
print("sigma_k für Zn: ", sigma_k)
print("E_abs für Zn : ", E_absorbzn)

#Zr
Z = 40
theta2, imp = np.genfromtxt("content/data/Zr40.txt", unpack = True)
plt.subplot(3, 2, 5)
x = np.linspace(20,21,1000)
plt.plot(theta2, imp, color = "firebrick", label = "Zr40")
plt.plot(20.5307, 59.5, "x", color = "cornflowerblue", label = r"$E_{\text{abs}}$")

plt.grid()
plt.xlabel(r'$2\theta \mathbin{/} \unit{\degree}$')
plt.ylabel(r'$\symup{Imp}\mathbin{/}\symup{s} $')
plt.legend()
plt.tight_layout()
plt.savefig('build/absorption.pdf')
plt.show()
plt.close()

imp_mittel = imp[0] + (imp[-1] - imp[0])/2 

E_absorbzr = const.h*const.c/(2*d_lif*np.sin(20.5307*np.pi/(2*180))*const.e*1000)
sigma_k = sigmak(Z, E_absorbzr)
print("sigma_k für Zr: ", sigma_k)
print("E_abs für Zr: ", E_absorbzr)

#Moseley

EHS = [np.sqrt(E_absorbzn*1000), np.sqrt(E_absorbga*1000), np.sqrt(E_absorbbr*1000),  np.sqrt(E_absorbsr*1000),  np.sqrt(E_absorbzr*1000)]
ZETS = [30, 31, 35, 38, 40]

#FIT
def linfit(x,m,b):
    return m*x+b

params, pcov = op.curve_fit(linfit, ZETS, EHS)

print("Rydbergkonstante nach dem Fit: ", params[0]**2)
x = np.linspace(29.5, 40.5, 1000)

plt.plot(ZETS, EHS,"x", color = "firebrick", label = "Mosley")
plt.plot(x, linfit(x, *params), color = "cornflowerblue", label = "Fit")
plt.tight_layout()
plt.grid()
plt.xlabel(r'$\text{Z}$')
plt.ylabel(r'$\sqrt{E_\text{abs}}\mathbin{/}\sqrt{\symup{eV}} $')
plt.legend()
#plt.savefig('build/plotbragg.pdf')
plt.show()
plt.close()
