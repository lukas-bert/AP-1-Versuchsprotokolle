import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
import scipy as sp
import uncertainties.unumpy as unp

#Einlesen der Daten
v_t1v175, v_t2v175, v_t3v175, v_t4v175, v_t5v175 = np.genfromtxt("content/data/175V.txt", unpack = True)
v_t1v200, v_t2v200, v_t3v200, v_t4v200, v_t5v200 = np.genfromtxt("content/data/200V.txt", unpack = True)
v_t1v225, v_t2v225, v_t3v225, v_t4v225, v_t5v225 = np.genfromtxt("content/data/225V.txt", unpack = True)
v_t1v250, v_t2v250, v_t3v250, v_t4v250, v_t5v250 = np.genfromtxt("content/data/250V.txt", unpack = True)
v_t1v275, v_t2v275, v_t3v275, v_t4v275, v_t5v275 = np.genfromtxt("content/data/275V.txt", unpack = True)

#Ergänzung einiger v_0:
#Messreihe 1:

v_01v175 = 0.1*10**(-3)/6.3 #m/s
v_02v175 = 0.5*10**(-3)/27.9

#Messreihe 2:

v_01v200 = 0.2*10**(-3)/24.35
v_02v200 = 0.2*10**(-3)/33.09
v_03v200 = 0.5*10**(-3)/26.44
v_04v200 = 0.5*10**(-3)/20.00
v_05v200 = 0.5*10**(-3)/25.30

#Berechnung der geschwindigkeiten:


v_t1v175 = 0.5*10**(-3)/v_t1v175
v_t2v175 = 0.5*10**(-3)/v_t2v175 
v_t3v175 = 0.5*10**(-3)/v_t3v175 
v_t4v175 = 0.5*10**(-3)/v_t4v175 
v_t5v175 = 0.5*10**(-3)/v_t5v175
v_t1v200 = 0.5*10**(-3)/v_t1v200 
v_t2v200 = 0.5*10**(-3)/v_t2v200 
v_t3v200 = 0.5*10**(-3)/v_t3v200 
v_t4v200 = 0.5*10**(-3)/v_t4v200 
v_t5v200 = 0.5*10**(-3)/v_t5v200
v_t1v225 = 0.5*10**(-3)/v_t1v225 
v_t2v225 = 0.5*10**(-3)/v_t2v225 
v_t3v225 = 0.5*10**(-3)/v_t3v225 
v_t4v225 = 0.5*10**(-3)/v_t4v225 
v_t5v225 = 0.5*10**(-3)/v_t5v225
v_t1v250 = 0.5*10**(-3)/v_t1v250 
v_t2v250 = 0.5*10**(-3)/v_t2v250 
v_t3v250 = 0.5*10**(-3)/v_t3v250 
v_t4v250 = 0.5*10**(-3)/v_t4v250 
v_t5v250 = 0.5*10**(-3)/v_t5v250
v_t1v275 = 0.5*10**(-3)/v_t1v275 
v_t2v275 = 0.5*10**(-3)/v_t2v275 
v_t3v275 = 0.5*10**(-3)/v_t3v275 
v_t4v275 = 0.5*10**(-3)/v_t4v275 
v_t5v275 = 0.5*10**(-3)/v_t5v275


v_t1v175om = np.mean(v_t1v175[0:3])
temp = sem(v_t1v175[0:3])
v_t1v175um = np.mean(v_t1v175[3:6])
temp1 = sem(v_t1v175[3:6])
temp2 = [v_t1v175om, v_t1v175um, v_01v175]
temp1err = [temp, temp1, 0]
t1v175 = unp.uarray(temp2, temp1err)
print(t1v175)

v_t2v175om = np.mean(v_t2v175[0:3])
temp = sem(v_t2v175[0:3])
v_t2v175um = np.mean(v_t2v175[3:6])
temp1 = sem(v_t2v175[3:6])
temp2 = [v_t2v175om, v_t2v175um, v_02v175]
temp1err = [temp, temp1, 0]
t2v175 = unp.uarray(temp2, temp1err)
print(t2v175)

v_t3v175om = np.mean(v_t3v175[0:3])
temp = sem(v_t3v175[0:3])
v_t3v175um = np.mean(v_t3v175[3:6])
temp1 = sem(v_t3v175[3:6])
temp2 = [v_t3v175om, v_t3v175um, 0]
temp1err = [temp, temp1, 0]
t3v175 = unp.uarray(temp2, temp1err)
print(t3v175)

v_t4v175om = np.mean(v_t4v175[0:3])
temp = sem(v_t4v175[0:3])
v_t4v175um = np.mean(v_t4v175[3:6])
temp1 = sem(v_t4v175[3:6])
temp2 = [v_t4v175om, v_t4v175um, 0]
temp1err = [temp, temp1, 0]
t4v175 = unp.uarray(temp2, temp1err)
print(t4v175)

v_t5v175om = np.mean(v_t5v175[0:3])
temp = sem(v_t5v175[0:3])
v_t5v175um = np.mean(v_t5v175[3:6])
temp1 = sem(v_t5v175[3:6])
temp2 = [v_t5v175om, v_t5v175um, 0]
temp1err = [temp, temp1, 0]
t5v175 = unp.uarray(temp2, temp1err)
print(t5v175)

v_t1v200om = np.mean(v_t1v200[0:3])
temp = sem(v_t1v200[0:3])
v_t1v200um = np.mean(v_t1v200[3:6])
temp1 = sem(v_t1v200[3:6])
temp2 = [v_t1v200om, v_t1v200um, v_01v200]
temp1err = [temp, temp1, 0]
t1v200 = unp.uarray(temp2, temp1err)
print(t1v200)

v_t2v200om = np.mean(v_t2v200[0:3])
temp = sem(v_t2v200[0:3])
v_t2v200um = np.mean(v_t2v200[3:6])
temp1 = sem(v_t2v200[3:6])
temp2 = [v_t2v200om, v_t2v200um, v_02v200]
temp1err = [temp, temp1, 0]
t2v200 = unp.uarray(temp2, temp1err)
print(t2v200)

v_t3v200om = np.mean(v_t3v200[0:3])
temp = sem(v_t3v200[0:3])
v_t3v200um = np.mean(v_t3v200[3:6])
temp1 = sem(v_t3v200[3:6])
temp2 = [v_t3v200om, v_t3v200um, v_03v200]
temp1err = [temp, temp1, 0]
t3v200 = unp.uarray(temp2, temp1err)
print(t3v200)

v_t4v200om = np.mean(v_t4v200[0:3])
temp = sem(v_t4v200[0:3])
v_t4v200um = np.mean(v_t4v200[3:6])
temp1 = sem(v_t4v200[3:6])
temp2 = [v_t4v200om, v_t4v200um, v_04v200]
temp1err = [temp, temp1, 0]
t4v200 = unp.uarray(temp2, temp1err)
print(t4v200)

v_t5v200om = np.mean(v_t5v200[0:3])
temp = sem(v_t5v200[0:3])
v_t5v200um = np.mean(v_t5v200[3:6])
temp1 = sem(v_t5v200[3:6])
temp2 = [v_t5v200om, v_t5v200um, v_05v200]
temp1err = [temp, temp1, 0]
t5v200 = unp.uarray(temp2, temp1err)
print(t5v200)

v_t1v225om = np.mean(v_t1v225[0:3])
temp = sem(v_t1v225[0:3])
v_t1v225um = np.mean(v_t1v225[3:6])
temp1 = sem(v_t1v225[3:6])
temp2 = [v_t1v225om, v_t1v225um, v_t1v225[6]]
temp1err = [temp, temp1, 0]
t1v225 = unp.uarray(temp2, temp1err)
print(t1v225)

v_t2v225om = np.mean(v_t2v225[0:3])
temp = sem(v_t2v225[0:3])
v_t2v225um = np.mean(v_t2v225[3:6])
temp1 = sem(v_t2v225[3:6])
temp2 = [v_t2v225om, v_t2v225um, v_t1v225[6]]
temp1err = [temp, temp1, 0]
t2v225 = unp.uarray(temp2, temp1err)
print(t2v225)

v_t3v225om = np.mean(v_t3v225[0:3])
temp = sem(v_t3v225[0:3])
v_t3v225um = np.mean(v_t3v225[3:6])
temp1 = sem(v_t3v225[3:6])
temp2 = [v_t3v225om, v_t3v225um, v_t1v225[6]]
temp1err = [temp, temp1, 0]
t3v225 = unp.uarray(temp2, temp1err)
print(t3v225)

v_t4v225om = np.mean(v_t4v225[0:3])
temp = sem(v_t4v225[0:3])
v_t4v225um = np.mean(v_t4v225[3:6])
temp1 = sem(v_t4v225[3:6])
temp2 = [v_t4v225om, v_t4v225um, v_t1v225[6]]
temp1err = [temp, temp1, 0]
t4v225 = unp.uarray(temp2, temp1err)
print(t4v225)

v_t5v225om = np.mean(v_t5v225[0:3])
temp = sem(v_t5v225[0:3])
v_t5v225um = np.mean(v_t5v225[3:6])
temp1 = sem(v_t5v225[3:6])
temp2 = [v_t5v225om, v_t5v225um, v_t1v225[6]]
temp1err = [temp, temp1, 0]
t5v225 = unp.uarray(temp2, temp1err)
print(t5v225)

v_t1v250om = np.mean(v_t1v250[0:3])
temp = sem(v_t1v250[0:3])
v_t1v250um = np.mean(v_t1v250[3:6])
temp1 = sem(v_t1v250[3:6])
temp2 = [v_t1v250om, v_t1v250um, v_t1v250[6]]
temp1err = [temp, temp1, 0]
t1v250 = unp.uarray(temp2, temp1err)
print(t1v250)

v_t2v250om = np.mean(v_t2v250[0:3])
temp = sem(v_t2v250[0:3])
v_t2v250um = np.mean(v_t2v250[3:6])
temp1 = sem(v_t2v250[3:6])
temp2 = [v_t2v250om, v_t2v250um, v_t1v250[6]]
temp1err = [temp, temp1, 0]
t2v250 = unp.uarray(temp2, temp1err)
print(t2v250)

v_t3v250om = np.mean(v_t3v250[0:3])
temp = sem(v_t3v250[0:3])
v_t3v250um = np.mean(v_t3v250[3:6])
temp1 = sem(v_t3v250[3:6])
temp2 = [v_t3v250om, v_t3v250um, v_t1v250[6]]
temp1err = [temp, temp1, 0]
t3v250 = unp.uarray(temp2, temp1err)
print(t3v250)

v_t4v250om = np.mean(v_t4v250[0:3])
temp = sem(v_t4v250[0:3])
v_t4v250um = np.mean(v_t4v250[3:6])
temp1 = sem(v_t4v250[3:6])
temp2 = [v_t4v250om, v_t4v250um, v_t1v250[6]]
temp1err = [temp, temp1, 0]
t4v250 = unp.uarray(temp2, temp1err)
print(t4v250)

v_t5v250om = np.mean(v_t5v250[0:3])
temp = sem(v_t5v250[0:3])
v_t5v250um = np.mean(v_t5v250[3:6])
temp1 = sem(v_t5v250[3:6])
temp2 = [v_t5v250om, v_t5v250um, v_t1v250[6]]
temp1err = [temp, temp1, 0]
t5v250 = unp.uarray(temp2, temp1err)
print(t5v250)

v_t1v275om = np.mean(v_t1v275[0:3])
temp = sem(v_t1v275[0:3])
v_t1v275um = np.mean(v_t1v275[3:6])
temp1 = sem(v_t1v275[3:6])
temp2 = [v_t1v275om, v_t1v275um, v_t1v275[6]]
temp1err = [temp, temp1, 0]
t1v275 = unp.uarray(temp2, temp1err)
print(t1v275)

v_t2v275om = np.mean(v_t2v275[0:3])
temp = sem(v_t2v275[0:3])
v_t2v275um = np.mean(v_t2v275[3:6])
temp1 = sem(v_t2v275[3:6])
temp2 = [v_t2v275om, v_t2v275um, v_t1v275[6]]
temp1err = [temp, temp1, 0]
t2v275 = unp.uarray(temp2, temp1err)
print(t2v275)

v_t3v275om = np.mean(v_t3v275[0:3])
temp = sem(v_t3v275[0:3])
v_t3v275um = np.mean(v_t3v275[3:6])
temp1 = sem(v_t3v275[3:6])
temp2 = [v_t3v275om, v_t3v275um, v_t1v275[6]]
temp1err = [temp, temp1, 0]
t3v275 = unp.uarray(temp2, temp1err)
print(t3v275)

v_t4v275om = np.mean(v_t4v275[0:3])
temp = sem(v_t4v275[0:3])
v_t4v275um = np.mean(v_t4v275[3:6])
temp1 = sem(v_t4v275[3:6])
temp2 = [v_t4v275om, v_t4v275um, v_t1v275[6]]
temp1err = [temp, temp1, 0]
t4v275 = unp.uarray(temp2, temp1err)
print(t4v275)

v_t5v275om = np.mean(v_t5v275[0:3])
temp = sem(v_t5v275[0:3])
v_t5v275um = np.mean(v_t5v275[3:6])
temp1 = sem(v_t5v275[3:6])
temp2 = [v_t5v275om, v_t5v275um, v_t1v275[6]]
temp1err = [temp, temp1, 0]
t5v275 = unp.uarray(temp2, temp1err)
print(t5v275)

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
