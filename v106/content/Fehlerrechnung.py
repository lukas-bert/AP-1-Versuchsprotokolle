import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import uncertainties.unumpy as unp
from uncertainties import ufloat
import scipy.constants as const

g = const.g
l_1 = 0.284
l_2 = 0.784
dl = 0.001      # Fehler beim Ablesen der Längen
N = 10          # Messumfang

# Umrechnungsfunktionen für T und omega(w) & Fehler dazu
def funcw(T): 
    return 2* np.pi / T   

def funcdw(T, dT):
    return (2* np.pi / T**2) * dT  

def funcT(w): 
    return 2* np.pi / w    

def funcdT(w, dw):
    return (2* np.pi / w**2) * dw 

# Messung 1     w:= Wertearray
wT_l, wT_r, wT_p, wT_m, wT, wT_s = np.genfromtxt("data_1.txt", unpack = True)        

# l: linkes Pendel, r: rechtes Pendel, p: +, m: -, s: Schwebung
# Index t: Theoriewert

# Mittelwerte
T_l = np.sum(wT_l)/(5*N)       # /(5*N) weil N Messwerte & 5 Schwingungsperioden
T_r = np.sum(wT_r)/(5*N)
T_p = np.sum(wT_p)/(5*N)
T_m = np.sum(wT_m)/(5*N)
T = np.sum(wT)/(3*N)           # 3 Perioden gemessen
T_s = np.sum(wT_s)/N       # Nur eine Periode gemessen

# Fehler der Mittelwerte

dT_l = np.sqrt(1/(N*(N-1))*(np.sum((wT_l/5-T_l)**2)))       # Faktor 1/5 wichtig !!!!
dT_r = np.sqrt(1/(N*(N-1))*(np.sum((wT_r/5-T_r)**2)))
dT_p = np.sqrt(1/(N*(N-1))*(np.sum((wT_p/5-T_p)**2)))
dT_m = np.sqrt(1/(N*(N-1))*(np.sum((wT_m/5-T_m)**2)))
dT = np.sqrt(1/(N*(N-1))*(np.sum((wT/3-T)**2)))
dT_s = np.sqrt(1/(N*(N-1))*(np.sum((wT_s-T_s)**2)))

# Frequenzen und deren Fehler berechnen: (Nur w+, w- und ws sind gefragt)
w_p = funcw(T_p)
w_m = funcw(T_m)
w_s = funcw(T_s)
dw_p = funcdw(T_p, dT_p)
dw_m = funcdw(T_m, dT_m)
dw_s = funcdw(T_s, dT_s)

# ufloats erstellen (u: ufloat)
uw_p = ufloat(w_p, dw_p)
uw_m = ufloat(w_m, dw_m)
uw_s = ufloat(w_s, dw_s)

# Berechnen der Theoriewerte
# Kopplungskonstante K

K = (T_p**2 - T_m**2)/(T_p**2 + T_m**2)
dK = (4*T_p*T_m)/(T_p**2 + T_m**2)**2 * np.sqrt(T_m**2 * dT_p**2 + T_p**2 * dT_m**2)

#print(K, dK)

# Eigenfrequenzen w+ & w-
w_pt = np.sqrt(g/l_1)
dw_pt = 0.5 * np.sqrt(g/(l_1)**3 * dl**2)
uw_pt = ufloat(w_pt, dw_pt)

w_mt = np.sqrt((g + 2*K)/l_1)
dw_mt = 1/(np.sqrt(l_1)*(g + 2*K))*np.sqrt(dK**2 + (dl*(g + 2* K)/(2*l_1))**2)
uw_mt = ufloat(w_mt, dw_mt)

# Schwebungsfrequenz 
#T_st = funcT(w_pt)*funcT(w_mt)/(funcT(w_pt) - funcT(w_mt))
#w_st = funcw(T_st)
w_st = w_mt - w_pt          # Formel in Versuchsanleitung ist falsch
dw_st = np.sqrt(dw_pt**2 + dw_mt**2)
uw_st = ufloat(w_st, dw_st)

#print(uw_st)

##################### Ausgeben der Werte #####################
print("Messung 1 Experimentelle Werte:  ", r'w_+: ', '{0:.4f}'.format(uw_p), r'w_-: ', '{0:.4f}'.format(uw_m), r'w_s: ', '{0:.4f}'.format(uw_s))
print("Messung 1 Theoretische Werte:    ", r'w_+: ', '{0:.4f}'.format(uw_pt), r'w_-: ', '{0:.4f}'.format(uw_mt), r'w_s: ', '{0:.4f}'.format(uw_st))
print()
##############################################################

#######################################################################################################################################################
#######################################################################################################################################################
# Messung 2
#######################################################################################################################################################
#######################################################################################################################################################

wT_l, wT_r, wT_p, wT_m, wT, wT_s = np.genfromtxt("data_2.txt",unpack = True)

# Mittelwerte
T_l = np.sum(wT_l)/(5*N)       # /(5*N) weil N Messwerte & 5 Schwingungsperioden
T_r = np.sum(wT_r)/(5*N)
T_p = np.sum(wT_p)/(5*N)
T_m = np.sum(wT_m)/(5*N)
T = np.sum(wT)/(5*N)           
T_s = np.sum(wT_s)/N       # Nur eine Periode gemessen

# Fehler der Mittelwerte

dT_l = np.sqrt(1/(N*(N-1))*(np.sum((wT_l/5-T_l)**2)))       # Faktor 1/5 wichtig !!!!
dT_r = np.sqrt(1/(N*(N-1))*(np.sum((wT_r/5-T_r)**2)))
dT_p = np.sqrt(1/(N*(N-1))*(np.sum((wT_p/5-T_p)**2)))
dT_m = np.sqrt(1/(N*(N-1))*(np.sum((wT_m/5-T_m)**2)))
dT = np.sqrt(1/(N*(N-1))*(np.sum((wT/5-T)**2)))
dT_s = np.sqrt(1/(N*(N-1))*(np.sum((wT_s-T_s)**2)))

# Frequenzen und deren Fehler berechnen: (Nur w+, w- und ws sind gefragt)
w_p = funcw(T_p)
w_m = funcw(T_m)
w_s = funcw(T_s)
dw_p = funcdw(T_p, dT_p)
dw_m = funcdw(T_m, dT_m)
dw_s = funcdw(T_s, dT_s)

# ufloats erstellen (u: ufloat)
uw_p = ufloat(w_p, dw_p)
uw_m = ufloat(w_m, dw_m)
uw_s = ufloat(w_s, dw_s)

# Berechnen der Theoriewerte
# Kopplungskonstante K

K = (T_p**2 - T_m**2)/(T_p**2 + T_m**2)
dK = (4*T_p*T_m)/(T_p**2 + T_m**2)**2 * np.sqrt(T_m**2 * dT_p**2 + T_p**2 * dT_m**2)

#print(K, dK)

# Eigenfrequenzen w+ & w-
w_pt = np.sqrt(g/l_2)
dw_pt = 0.5 * np.sqrt(g/(l_2)**3 * dl**2)
uw_pt = ufloat(w_pt, dw_pt)

w_mt = np.sqrt((g + 2*K)/l_2)
dw_mt = 1/(np.sqrt(l_2)*(g + 2*K))*np.sqrt(dK**2 + (dl*(g + 2* K)/(2*l_2))**2)
uw_mt = ufloat(w_mt, dw_mt)

# Schwebungsfrequenz 
#T_st = funcT(w_pt)*funcT(w_mt)/(funcT(w_pt) - funcT(w_mt))
#w_st = funcw(T_st)
w_st = w_mt - w_pt          # Formel in Versuchsanleitung ist falsch
dw_st = np.sqrt(dw_pt**2 + dw_mt**2)
uw_st = ufloat(w_st, dw_st)

#print(uw_st)

##################### Ausgeben der Werte #####################
print("Messung 2 Experimentelle Werte:  ", r'w_+: ', '{0:.4f}'.format(uw_p), r'w_-: ', '{0:.4f}'.format(uw_m), r'w_s: ', '{0:.4f}'.format(uw_s))
print("Messung 2 Theoretische Werte:    ", r'w_+: ', '{0:.4f}'.format(uw_pt), r'w_-: ', '{0:.4f}'.format(uw_mt), r'w_s: ', '{0:.4f}'.format(uw_st))
##############################################################