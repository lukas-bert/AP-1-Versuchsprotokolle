import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
import scipy as sp
import uncertainties.unumpy as unp
import scipy.constants as const
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as devs
from uncertainties import ufloat

#Einlesen der Daten
v_t1v175, v_t2v175, v_t3v175, v_t4v175, v_t5v175 = np.genfromtxt("content/data/175V.txt", unpack = True)
v_t1v200, v_t2v200, v_t3v200, v_t4v200, v_t5v200 = np.genfromtxt("content/data/200V.txt", unpack = True)
v_t1v225, v_t2v225, v_t3v225, v_t4v225, v_t5v225 = np.genfromtxt("content/data/225V.txt", unpack = True)
v_t1v250, v_t2v250, v_t3v250, v_t4v250, v_t5v250 = np.genfromtxt("content/data/250V.txt", unpack = True)
v_t1v275, v_t2v275, v_t3v275, v_t4v275, v_t5v275 = np.genfromtxt("content/data/275V.txt", unpack = True)

#Ergänzung einiger v_0:
#Messreihe 1:

v_01v175 = 0.1*10**(-3)/6.3 #m/s
v_05v175 = 0.5*10**(-3)/27.9

#Messreihe 2:

v_01v200 = 0.2*10**(-3)/24.35
v_02v200 = 0.2*10**(-3)/33.09
v_03v200 = 0.5*10**(-3)/26.44
v_04v200 = 0.5*10**(-3)/20.00
v_05v200 = 0.5*10**(-3)/25.30

#Berechnung der Geschwindigkeiten:

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
temp2 = [v_t2v175om, v_t2v175um, 0]
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
temp2 = [v_t5v175om, v_t5v175um, v_05v175]
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
temp2 = [v_t2v225om, v_t2v225um, v_t2v225[6]]
temp1err = [temp, temp1, 0]
t2v225 = unp.uarray(temp2, temp1err)
print(t2v225)

v_t3v225om = np.mean(v_t3v225[0:3])
temp = sem(v_t3v225[0:3])
v_t3v225um = np.mean(v_t3v225[3:6])
temp1 = sem(v_t3v225[3:6])
temp2 = [v_t3v225om, v_t3v225um, v_t3v225[6]]
temp1err = [temp, temp1, 0]
t3v225 = unp.uarray(temp2, temp1err)
print(t3v225)

v_t4v225om = np.mean(v_t4v225[0:3])
temp = sem(v_t4v225[0:3])
v_t4v225um = np.mean(v_t4v225[3:6])
temp1 = sem(v_t4v225[3:6])
temp2 = [v_t4v225om, v_t4v225um, v_t4v225[6]]
temp1err = [temp, temp1, 0]
t4v225 = unp.uarray(temp2, temp1err)
print(t4v225)

v_t5v225om = np.mean(v_t5v225[0:3])
temp = sem(v_t5v225[0:3])
v_t5v225um = np.mean(v_t5v225[3:6])
temp1 = sem(v_t5v225[3:6])
temp2 = [v_t5v225om, v_t5v225um, v_t5v225[6]]
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
temp2 = [v_t2v250om, v_t2v250um, v_t2v250[6]]
temp1err = [temp, temp1, 0]
t2v250 = unp.uarray(temp2, temp1err)
print(t2v250)

v_t3v250om = np.mean(v_t3v250[0:3])
temp = sem(v_t3v250[0:3])
v_t3v250um = np.mean(v_t3v250[3:6])
temp1 = sem(v_t3v250[3:6])
temp2 = [v_t3v250om, v_t3v250um, v_t3v250[6]]
temp1err = [temp, temp1, 0]
t3v250 = unp.uarray(temp2, temp1err)
print(t3v250)

v_t4v250om = np.mean(v_t4v250[0:3])
temp = sem(v_t4v250[0:3])
v_t4v250um = np.mean(v_t4v250[3:6])
temp1 = sem(v_t4v250[3:6])
temp2 = [v_t4v250om, v_t4v250um, v_t4v250[6]]
temp1err = [temp, temp1, 0]
t4v250 = unp.uarray(temp2, temp1err)
print(t4v250)

v_t5v250om = np.mean(v_t5v250[0:3])
temp = sem(v_t5v250[0:3])
v_t5v250um = np.mean(v_t5v250[3:6])
temp1 = sem(v_t5v250[3:6])
temp2 = [v_t5v250om, v_t5v250um, v_t5v250[6]]
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
temp2 = [v_t2v275om, v_t2v275um, v_t2v275[6]]
temp1err = [temp, temp1, 0]
t2v275 = unp.uarray(temp2, temp1err)
print(t2v275)

v_t3v275om = np.mean(v_t3v275[0:3])
temp = sem(v_t3v275[0:3])
v_t3v275um = np.mean(v_t3v275[3:6])
temp1 = sem(v_t3v275[3:6])
temp2 = [v_t3v275om, v_t3v275um, v_t3v275[6]]
temp1err = [temp, temp1, 0]
t3v275 = unp.uarray(temp2, temp1err)
print(t3v275)

v_t4v275om = np.mean(v_t4v275[0:3])
temp = sem(v_t4v275[0:3])
v_t4v275um = np.mean(v_t4v275[3:6])
temp1 = sem(v_t4v275[3:6])
temp2 = [v_t4v275om, v_t4v275um, v_t4v275[6]]
temp1err = [temp, temp1, 0]
t4v275 = unp.uarray(temp2, temp1err)
print(t4v275)

v_t5v275om = np.mean(v_t5v275[0:3])
temp = sem(v_t5v275[0:3])
v_t5v275um = np.mean(v_t5v275[3:6])
temp1 = sem(v_t5v275[3:6])
temp2 = [v_t5v275om, v_t5v275um, v_t5v275[6]]
temp1err = [temp, temp1, 0]
t5v275 = unp.uarray(temp2, temp1err)
print(t5v275)

############################################################################################################################
############################################################################################################################
############################################################################################################################

# Konstanten
g = const.g
d = ufloat(7.6250, 0.0051)*10**(-3)     # Abstand der Kondensatorplatten
p_oel = 886     # kg/m^3 
p_L = 1.204  # kg/m^3 bei 20°C
B = 6.17*10**(-5)*133.322               # B in Pa*m  
e = const.e          

def Radius(v_0, n_L):        # beachte: 2*v_o = v_ab - v_auf
    return unp.sqrt((9*n_L*v_0)/(2*g*(p_oel- p_L)))

def Ladung(v_ab, v_auf, U, v_0, n_L):
    return 9/2*np.pi*unp.sqrt((n_L**3*(v_ab-v_auf))/(g*(p_oel-p_L))) * d*(v_ab + v_auf)/U

# Korrektur der viskosität nach cunningham:
def eta_eff(v_0, n_L):
    return n_L*(1/(1 + B/(100000*Radius(v_0, n_L))))    

# Überprüfung der Bedingung 2v_0 = v_ab - v_auf

def test(v_ab, v_0, v_auf, tol, name):
    err = np.abs(2*noms(v_0)-(noms(v_ab)-noms(v_auf)))/(noms(v_0)*2)
    if(err > tol):
        print("Die Bedingung 2v_0 = v_ab - v_auf ist mit einer Toleranz von: ", tol, " für ", name, " nicht erfüllt (Abweichung: ", err , ")")
    return

# Korrektur der Ladung nach Cunnningham:

def q_real(v_ab, v_auf, U, v_0, n_L):
    return Ladung(v_ab, v_auf, U, v_0, n_L)*(1 + (B/(100000*Radius((v_ab - v_auf)/2, n_L))))**(3/2)

# Ausführen auf Messreihen zuerst U = 175V und U = 200V, nicht immer v_o vorhanden :():

# tXvYyy : Teilchen X, Spannung Yyy Volt 
# tXvYyy = [v_auf, v_ab, v_0] mit Fehlern

test(t1v175[1], t1v175[2], t1v175[0], 0, "T1, U = 175V")
test(t5v175[1], t5v175[2], t5v175[0], 0, "T5, U = 175V")

test(t1v200[1], t1v200[2], t1v200[0], 0, "T1, U = 200V")
test(t2v200[1], t2v200[2], t2v200[0], 0, "T2, U = 200V")
test(t3v200[1], t3v200[2], t3v200[0], 0, "T3, U = 200V")
test(t4v200[1], t4v200[2], t4v200[0], 0, "T4, U = 200V")
test(t5v200[1], t5v200[2], t5v200[0], 0, "T5, U = 200V")

test(t1v225[1], t1v225[2], t1v225[0], 0, "T1, U = 225V")
test(t2v225[1], t2v225[2], t2v225[0], 0, "T2, U = 225V")
test(t3v225[1], t3v225[2], t3v225[0], 0, "T3, U = 225V")
test(t4v225[1], t4v225[2], t4v225[0], 0, "T4, U = 225V")
test(t5v225[1], t5v225[2], t5v225[0], 0, "T5, U = 225V")

test(t1v250[1], t1v250[2], t1v250[0], 0, "T1, U = 250V")
test(t2v250[1], t2v250[2], t2v250[0], 0, "T2, U = 250V")
test(t3v250[1], t3v250[2], t3v250[0], 0, "T3, U = 250V")
test(t4v250[1], t4v250[2], t4v250[0], 0, "T4, U = 250V")
test(t5v250[1], t5v250[2], t5v250[0], 0, "T5, U = 250V")

test(t1v275[1], t1v275[2], t1v275[0], 0, "T1, U = 275V")
test(t2v275[1], t2v275[2], t2v275[0], 0, "T2, U = 275V")
test(t3v275[1], t3v275[2], t3v275[0], 0, "T3, U = 275V")
test(t4v275[1], t4v275[2], t4v275[0], 0, "T4, U = 275V")
test(t5v275[1], t5v275[2], t5v275[0], 0, "T5, U = 275V")


#Temperatur zu der Messreihe v175:
T_v175   = 295.15 # Kelvin
T_v175_5 = 303.15 # Kelvin
n_L175 = 1.833e-5
n_L175_5 = 1.8715e-5

#Temperatur zu der Messreihe v200:
T_v200_12 = 295.15 # Kelvin
T_v200 = 302.15 # Kelvin
n_L200_12 = 1.833e-5
n_L200 = 1.866e-5

#Temperatur zu der Messreihe v225:
T_v225 = 300.15 # Kelvin
n_L225 = 1.867e-5

#Temperatur zu der Messreihe v250:
T_v250 = 298.15 # Kelvin
n_L250 = 1.848e-5

#Temperatur zu der Messreihe v275:
T_v275 = 303.15 # Kelvin
n_L275 = 1.8715e-5
n_L275_eff = eta_eff(1/2*(t5v275[1] - t5v275[0]), n_L275)

#######################################################################



ladungenv175 = unp.uarray([noms(q_real(t1v175[1], t1v175[0], 175, t1v175[2], n_L175)),
noms(q_real(t2v175[1], t2v175[0], 175, t2v175[2], n_L175)),
noms(q_real(t3v175[1], t3v175[0], 175, t3v175[2], n_L175)),
noms(q_real(t4v175[1], t4v175[0], 175, t4v175[2], n_L175)),
noms(q_real(t5v175[1], t5v175[0], 175, t5v175[2], n_L175_5))],[devs(q_real(t1v175[1], t1v175[0], 175, t1v175[2], n_L175)),
devs(q_real(t2v175[1], t2v175[0], 175, t2v175[2], n_L175)),
devs(q_real(t3v175[1], t3v175[0], 175, t3v175[2], n_L175)),
devs(q_real(t4v175[1], t4v175[0], 175, t4v175[2], n_L175)),
devs(q_real(t5v175[1], t5v175[0], 175, t5v175[2], n_L175_5))])




ladungenv200 = unp.uarray([noms(q_real(t1v200[1], t1v200[0], 200, t1v200[2], n_L200_12)),
noms(q_real(t2v200[1], t2v200[0], 200, t2v200[2], n_L200_12)),
noms(q_real(t3v200[1], t3v200[0], 200, t3v200[2], n_L200)),
noms(q_real(t4v200[1], t4v200[0], 200, t4v200[2], n_L200)),
noms(q_real(t5v200[1], t5v200[0], 200, t5v200[2], n_L200))],[devs(q_real(t1v200[1], t1v200[0], 200, t1v200[2], n_L200_12)),
devs(q_real(t2v200[1], t2v200[0], 200, t2v200[2], n_L200_12)),
devs(q_real(t3v200[1], t3v200[0], 200, t3v200[2], n_L200)),
devs(q_real(t4v200[1], t4v200[0], 200, t4v200[2], n_L200)),
devs(q_real(t5v200[1], t5v200[0], 200, t5v200[2], n_L200))])




ladungenv225 = unp.uarray([noms(q_real(t1v225[1], t1v225[0], 225, t1v225[2], n_L225)),
noms(q_real(t2v225[1], t2v225[0], 225, t2v225[2], n_L225)),
noms(q_real(t3v225[1], t3v225[0], 225, t3v225[2], n_L225)),
noms(q_real(t4v225[1], t4v225[0], 225, t4v225[2], n_L225)),
noms(q_real(t5v225[1], t5v225[0], 225, t5v225[2], n_L225))],[devs(q_real(t1v225[1], t1v225[0], 225, t1v225[2], n_L225)),
noms(q_real(t2v225[1], t2v225[0], 225, t2v225[2], n_L225)),
devs(q_real(t3v225[1], t3v225[0], 225, t3v225[2], n_L225)),
devs(q_real(t4v225[1], t4v225[0], 225, t4v225[2], n_L225)),
devs(q_real(t5v225[1], t5v225[0], 225, t5v225[2], n_L225))])


ladungenv250 = unp.uarray([noms(q_real(t1v250[1], t1v250[0], 250, t1v250[2], n_L250)),
noms(q_real(t2v250[1], t2v250[0], 250, t2v250[2], n_L250)),
noms(q_real(t3v250[1], t3v250[0], 250, t3v250[2], n_L250)),
noms(q_real(t4v250[1], t4v250[0], 250, t4v250[2], n_L250)),
noms(q_real(t5v250[1], t5v250[0], 250, t5v250[2], n_L250))],[devs(q_real(t1v250[1], t1v250[0], 250, t1v250[2], n_L250)),
devs(q_real(t2v250[1], t2v250[0], 250, t2v250[2], n_L250)),
devs(q_real(t3v250[1], t3v250[0], 250, t3v250[2], n_L250)),
devs(q_real(t4v250[1], t4v250[0], 250, t4v250[2], n_L250)),
devs(q_real(t5v250[1], t5v250[0], 250, t5v250[2], n_L250))])


ladungenv275 = unp.uarray([noms(q_real(t1v275[1], t1v275[0], 275, t1v275[2], n_L275)),
noms(q_real(t2v275[1], t2v275[0], 275, t2v275[2], n_L275)),
noms(q_real(t3v275[1], t3v275[0], 275, t3v275[2], n_L275)),
noms(q_real(t4v275[1], t4v275[0], 275, t4v275[2], n_L275)),
noms(q_real(t5v275[1], t5v275[0], 275, t5v275[2], n_L275))],[devs(q_real(t1v275[1], t1v275[0], 275, t1v275[2], n_L275)),
devs(q_real(t2v275[1], t2v275[0], 275, t2v275[2], n_L275)),
devs(q_real(t3v275[1], t3v275[0], 275, t3v275[2], n_L275)),
devs(q_real(t4v275[1], t4v275[0], 275, t4v275[2], n_L275)),
devs(q_real(t5v275[1], t5v275[0], 275, t5v275[2], n_L275))])

print(ladungenv175)
print(ladungenv200)
print(ladungenv225)
print(ladungenv250)
print(ladungenv275)

# Bestimmung der Elementarladung aus den Messdaten

def elementar(arr):
    e_0 = 999
    for i in range(len(arr)):
        for j in (np.array(range(len(ladungenv175)-1)) + 1):
            a = np.abs(arr[i] - arr[j])
            if a > 1e-19 and a < e_0:
                e_0 = a
    return e_0       

e_0v175 = elementar(ladungenv175)
e_0v200 = elementar(ladungenv200)
e_0v225 = elementar(ladungenv225)
e_0v250 = elementar(ladungenv250)
e_0v275 = elementar(ladungenv275)

print("--------------------------------------------------------------------------------------------------")
print("korrigierte Ladungen")
print("Elementarladung aus V = 175V: ", '{0:.3e}'.format(e_0v175))
print("Elementarladung aus V = 200V: ", '{0:.3e}'.format(e_0v200))
print("Elementarladung aus V = 225V: ", '{0:.3e}'.format(e_0v225))
print("Elementarladung aus V = 250V: ", '{0:.3e}'.format(e_0v250))
print("Elementarladung aus V = 275V: ", '{0:.3e}'.format(e_0v275))
print("--------------------------------------------------------------------------------------------------")

##bestimmung der e_0 aus den Messwerten v175 (dubios)
#eladungenv175 = np.around(noms(ladungenv175)/const.e)
##print(eladungenv175)
#
#arraye_0v175 = ladungenv175/eladungenv175 
##print(arraye_0v175)
#e_0v175 = np.mean(arraye_0v175)
#print(e_0v175)
#
#
##bestimmung der e_0 aus den Messwerten v200 (dubios)
#eladungenv200 = np.around(noms(ladungenv200)/const.e)
##print(eladungenv200)
#
#arraye_0v200 = ladungenv200/eladungenv200 
##print(arraye_0v200)
#e_0v200 = np.mean(arraye_0v200)
#print(e_0v200)
#
##bestimmung der e_0 aus den Messwerten v225 (dubios)
#eladungenv225 = np.around(noms(ladungenv225)/const.e)
##print(eladungenv225)
#
#arraye_0v225 = ladungenv225/eladungenv225 
##print(arraye_0v225)
#e_0v225 = np.mean(arraye_0v225)
#print(e_0v225)
#
#
##bestimmung der e_0 aus den Messwerten v250 (dubios)
#eladungenv250 = np.around(noms(ladungenv250)/const.e)
##print(eladungenv250)
#
#arraye_0v250 = ladungenv250/eladungenv250 
##print(arraye_0v250)
#e_0v250 = np.mean(arraye_0v250)
#print(e_0v250)
#
##bestimmung der e_0 aus den Messwerten v275 (dubios)
#eladungenv275 = np.around(noms(ladungenv275)/const.e)
##print(eladungenv275)
#
#arraye_0v275 = ladungenv275/eladungenv275 
##print(arraye_0v275)
#e_0v275 = np.mean(arraye_0v275)
#print(e_0v275)

#Mittelwert der e_0s
print("--------------------------------------------------------------------------------------------------")
e_0m = (e_0v175 + e_0v200 + e_0v225 + e_0v250 + e_0v275)/5
print("Mittelwert: ", '{0:.3e}'.format(e_0m))

#Abweichung zu lit
deltae_0m = np.abs(e_0m - const.e)/const.e
print("Abweichung e zum Literaturwert: ", '{0:.4f}'.format(deltae_0m))
print("--------------------------------------------------------------------------------------------------")

# Avogadrokonstante

F = ufloat(96485.3399, 0.0024)
N_A = F/e_0m
dN_A = np.abs(noms(N_A)- const.N_A)/(const.N_A)
print("Avogadrokonstante: ", '{0:.4e}'.format(N_A))
print("Abweichung: ", '{0:.4f}'.format(dN_A))

# Plot zur Ladungsverteilung
x = np.array([0.9, 0.95, 1, 1.05, 1.1])
plt.errorbar(x,   noms(ladungenv175), yerr = devs(ladungenv175), elinewidth = 0.7, linewidth = 0, marker = ".", markersize = 7, capsize=3)
plt.errorbar(1+x, noms(ladungenv200), yerr = devs(ladungenv200), elinewidth = 0.7, linewidth = 0, marker = ".", markersize = 7, capsize=3)
plt.errorbar(2+x, noms(ladungenv225), yerr = devs(ladungenv225), elinewidth = 0.7, linewidth = 0, marker = ".", markersize = 7, capsize=3)
plt.errorbar(3+x, noms(ladungenv250), yerr = devs(ladungenv250), elinewidth = 0.7, linewidth = 0, marker = ".", markersize = 7, capsize=3)
plt.errorbar(4+x, noms(ladungenv275), yerr = devs(ladungenv275), elinewidth = 0.7, linewidth = 0, marker = ".", markersize = 7, capsize=3)

plt.grid()
plt.yticks([0, e, 2*e, 3*e, 4*e, 5*e, 6*e, 7*e, 10*e, 12*e], [0, r"$e$", r"$2e$", r"$3e$", r"$4e$", r"$5e$", r"$6e$", r"$7e$", r"$10e$", r"$12e$"])
plt.xticks([1, 2, 3, 4, 5], [r"$175 \unit{\volt}$", r"$200 \unit{\volt}$", r"$225 \unit{\volt}$", r"$250 \unit{\volt}$", r"$275 \unit{\volt}$"])
plt.ylim(0*e, 13*e)
plt.tight_layout()

plt.show()
plt.savefig("build/plot.pdf")


########################### METHODE 2 #########################################################
print("methode 2")

ladungenv175 = unp.uarray([noms(q_real(t1v175[1], t1v175[0], 175, t1v175[2], n_L175)),
noms(Ladung(t2v175[1], t2v175[0], 175, t2v175[2], n_L175)),
noms(Ladung(t3v175[1], t3v175[0], 175, t3v175[2], n_L175)),
noms(Ladung(t4v175[1], t4v175[0], 175, t4v175[2], n_L175)),
noms(Ladung(t5v175[1], t5v175[0], 175, t5v175[2], n_L175_5))],[devs(Ladung(t1v175[1], t1v175[0], 175, t1v175[2], n_L175)),
devs(Ladung(t2v175[1], t2v175[0], 175, t2v175[2], n_L175)),
devs(Ladung(t3v175[1], t3v175[0], 175, t3v175[2], n_L175)),
devs(Ladung(t4v175[1], t4v175[0], 175, t4v175[2], n_L175)),
devs(Ladung(t5v175[1], t5v175[0], 175, t5v175[2], n_L175_5))])


ladungenv200 = unp.uarray([noms(q_real(t1v200[1], t1v200[0], 200, t1v200[2], n_L200_12)),
noms(Ladung(t2v200[1], t2v200[0], 200, t2v200[2], n_L200_12)),
noms(Ladung(t3v200[1], t3v200[0], 200, t3v200[2], n_L200)),
noms(Ladung(t4v200[1], t4v200[0], 200, t4v200[2], n_L200)),
noms(Ladung(t5v200[1], t5v200[0], 200, t5v200[2], n_L200))],[devs(Ladung(t1v200[1], t1v200[0], 200, t1v200[2], n_L200_12)),
devs(Ladung(t2v200[1], t2v200[0], 200, t2v200[2], n_L200_12)),
devs(Ladung(t3v200[1], t3v200[0], 200, t3v200[2], n_L200)),
devs(Ladung(t4v200[1], t4v200[0], 200, t4v200[2], n_L200)),
devs(Ladung(t5v200[1], t5v200[0], 200, t5v200[2], n_L200))])


ladungenv225 = unp.uarray([noms(q_real(t1v225[1], t1v225[0], 225, t1v225[2], n_L225)),
noms(Ladung(t2v225[1], t2v225[0], 225, t2v225[2], n_L225)),
noms(Ladung(t3v225[1], t3v225[0], 225, t3v225[2], n_L225)),
noms(Ladung(t4v225[1], t4v225[0], 225, t4v225[2], n_L225)),
noms(Ladung(t5v225[1], t5v225[0], 225, t5v225[2], n_L225))],[devs(Ladung(t1v225[1], t1v225[0], 225, t1v225[2], n_L225)),
noms(Ladung(t2v225[1], t2v225[0], 225, t2v225[2], n_L225)),
devs(Ladung(t3v225[1], t3v225[0], 225, t3v225[2], n_L225)),
devs(Ladung(t4v225[1], t4v225[0], 225, t4v225[2], n_L225)),
devs(Ladung(t5v225[1], t5v225[0], 225, t5v225[2], n_L225))])


ladungenv250 = unp.uarray([noms(q_real(t1v250[1], t1v250[0], 250, t1v250[2], n_L250)),
noms(Ladung(t2v250[1], t2v250[0], 250, t2v250[2], n_L250)),
noms(Ladung(t3v250[1], t3v250[0], 250, t3v250[2], n_L250)),
noms(Ladung(t4v250[1], t4v250[0], 250, t4v250[2], n_L250)),
noms(Ladung(t5v250[1], t5v250[0], 250, t5v250[2], n_L250))],[devs(Ladung(t1v250[1], t1v250[0], 250, t1v250[2], n_L250)),
devs(Ladung(t2v250[1], t2v250[0], 250, t2v250[2], n_L250)),
devs(Ladung(t3v250[1], t3v250[0], 250, t3v250[2], n_L250)),
devs(Ladung(t4v250[1], t4v250[0], 250, t4v250[2], n_L250)),
devs(Ladung(t5v250[1], t5v250[0], 250, t5v250[2], n_L250))])


ladungenv275 = unp.uarray([noms(q_real(t1v275[1], t1v275[0], 275, t1v275[2], n_L275)),
noms(Ladung(t2v275[1], t2v275[0], 275, t2v275[2], n_L275)),
noms(Ladung(t3v275[1], t3v275[0], 275, t3v275[2], n_L275)),
noms(Ladung(t4v275[1], t4v275[0], 275, t4v275[2], n_L275)),
noms(Ladung(t5v275[1], t5v275[0], 275, t5v275[2], n_L275))],[devs(Ladung(t1v275[1], t1v275[0], 275, t1v275[2], n_L275)),
devs(Ladung(t2v275[1], t2v275[0], 275, t2v275[2], n_L275)),
devs(Ladung(t3v275[1], t3v275[0], 275, t3v275[2], n_L275)),
devs(Ladung(t4v275[1], t4v275[0], 275, t4v275[2], n_L275)),
devs(Ladung(t5v275[1], t5v275[0], 275, t5v275[2], n_L275))])

e_0v175 = elementar(ladungenv175)
e_0v200 = elementar(ladungenv200)
e_0v225 = elementar(ladungenv225)
e_0v250 = elementar(ladungenv250)
e_0v275 = elementar(ladungenv275)

print("--------------------------------------------------------------------------------------------------")
print("unkorrigierte Ladungen")
print("Elementarladung aus V = 175V: ", '{0:.3e}'.format(e_0v175))
print("Elementarladung aus V = 200V: ", '{0:.3e}'.format(e_0v200))
print("Elementarladung aus V = 225V: ", '{0:.3e}'.format(e_0v225))
print("Elementarladung aus V = 250V: ", '{0:.3e}'.format(e_0v250))
print("Elementarladung aus V = 275V: ", '{0:.3e}'.format(e_0v275))
print("--------------------------------------------------------------------------------------------------")

#Mittelwert der e_0s
print("--------------------------------------------------------------------------------------------------")
e_0m = (e_0v175 + e_0v200 + e_0v225 + e_0v250 + e_0v275)/5
print("Mittelwert: ", '{0:.3e}'.format(e_0m))

#Abweichung zu lit
deltae_0m = np.abs(e_0m - const.e)/const.e
print("Abweichung e zum Literaturwert: ", '{0:.4f}'.format(deltae_0m))
print("--------------------------------------------------------------------------------------------------")

##bestimmung der e_0 aus den Messwerten v175 (dubios)
#eladungenv175 = np.around(noms(ladungenv175)/const.e)
##print(eladungenv175)
#
#arraye_0v175 = ladungenv175/eladungenv175 
##print(arraye_0v175)
#e_0v175 = np.mean(arraye_0v175)
#print(e_0v175)
#
#
##bestimmung der e_0 aus den Messwerten v200 (dubios)
#eladungenv200 = np.around(noms(ladungenv200)/const.e)
##print(eladungenv200)
#
#arraye_0v200 = ladungenv200/eladungenv200 
##print(arraye_0v200)
#e_0v200 = np.mean(arraye_0v200)
#print(e_0v200)
#
##bestimmung der e_0 aus den Messwerten v225 (dubios)
#eladungenv225 = np.around(noms(ladungenv225)/const.e)
##print(eladungenv225)
#
#arraye_0v225 = ladungenv225/eladungenv225 
##print(arraye_0v225)
#e_0v225 = np.mean(arraye_0v225)
#print(e_0v225)
#
#
##bestimmung der e_0 aus den Messwerten v250 (dubios)
#eladungenv250 = np.around(noms(ladungenv250)/const.e)
##print(eladungenv250)
#
#arraye_0v250 = ladungenv250/eladungenv250 
##print(arraye_0v250)
#e_0v250 = np.mean(arraye_0v250)
#print(e_0v250)
#
##bestimmung der e_0 aus den Messwerten v275 (dubios)
#eladungenv275 = np.around(noms(ladungenv275)/const.e)
##print(eladungenv275)
#
#arraye_0v275 = ladungenv275/eladungenv275 
##print(arraye_0v275)
#e_0v275 = np.mean(arraye_0v275)
#print(e_0v275)
#
##Mittelwert der e_0s
#e_0m = (e_0v175 + e_0v200 + e_0v225 + e_0v250 + e_0v275)/5
#print(e_0m)
#
##Abweichung zu lit
#deltae_0m = np.abs(e_0m - const.e)*100/const.e
#print(deltae_0m)

# Print der Radien für Tabelle

#print(
# Radius((t1v175[1] - t1v175[0])/2, n_L175),
# Radius((t2v175[1] - t2v175[0])/2, n_L175),
# Radius((t3v175[1] - t3v175[0])/2, n_L175),
# Radius((t4v175[1] - t4v175[0])/2, n_L175),
# Radius((t5v175[1] - t5v175[0])/2, n_L175_5))
#print(
# Radius((t1v200[1] - t1v200[0])/2, n_L200_12),
# Radius((t2v200[1] - t2v200[0])/2, n_L200_12),
# Radius((t3v200[1] - t3v200[0])/2, n_L200),
# Radius((t4v200[1] - t4v200[0])/2, n_L200),
# Radius((t5v200[1] - t5v200[0])/2, n_L200)
#)
#print(
# Radius((t1v225[1] - t1v225[0])/2, n_L225),
# Radius((t2v225[1] - t2v225[0])/2, n_L225),
# Radius((t3v225[1] - t3v225[0])/2, n_L225),
# Radius((t4v225[1] - t4v225[0])/2, n_L225),
# Radius((t5v225[1] - t5v225[0])/2, n_L225)
#)
#print(
# Radius((t1v250[1] - t1v250[0])/2, n_L250),
# Radius((t2v250[1] - t2v250[0])/2, n_L250),
# Radius((t3v250[1] - t3v250[0])/2, n_L250),
# Radius((t4v250[1] - t4v250[0])/2, n_L250),
# Radius((t5v250[1] - t5v250[0])/2, n_L250)
#)
#print(
# Radius((t1v275[1] - t1v275[0])/2, n_L275),
# Radius((t2v275[1] - t2v275[0])/2, n_L275),
# Radius((t3v275[1] - t3v275[0])/2, n_L275),
# Radius((t4v275[1] - t4v275[0])/2, n_L275),
# Radius((t5v275[1] - t5v275[0])/2, n_L275)
#)
#
#print(ladungenv175)
#print(ladungenv200)
#print(ladungenv225)
#print(ladungenv250)
#print(ladungenv275)
#