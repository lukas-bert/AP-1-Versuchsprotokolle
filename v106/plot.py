import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import uncertainties.unumpy as unp
from uncertainties import ufloat
import scipy.constants as const
l1 = ufloat(28.4,0.1)
T_l ,T_r ,T_p ,T_m ,T , T_s = np.genfromtxt("content/data_1.txt",unpack = True)
l2 = ufloat(78.4,0.1)
T_l2 ,T_r2 ,T_p2 ,T_m2 ,T2 , T_s2 = np.genfromtxt("content/data_2.txt",unpack = True)
#Fehlerrechnung mit uncertainties
fT_l = unp.uarray(T_l,0.25)
fT_r = unp.uarray(T_r,0.25)
fT_p = unp.uarray(T_p,0.25)
fT_m = unp.uarray(T_m,0.25)
fT = unp.uarray(T,0.25) 
fT_s = unp.uarray(T_s,0.25)

fT_l2 = unp.uarray(T_l2,0.25)
fT_r2 = unp.uarray(T_r2,0.25)
fT_p2 = unp.uarray(T_p2,0.25)
fT_m2 = unp.uarray(T_m2,0.25)
fT2 = unp.uarray(T2,0.25) 
fT_s2 = unp.uarray(T_s2,0.25)

MfT_l = np.mean(fT_l)
MfT_r = np.mean(fT_r)
MfT_p = np.mean(fT_p)
MfT_m = np.mean(fT_m)
MfT = np.mean(fT)
MfT_s = np.mean(fT_s)

MfT_l2 = np.mean(fT_l2)
MfT_r2 = np.mean(fT_r2)
MfT_p2 = np.mean(fT_p2)
MfT_m2 = np.mean(fT_m2)
MfT2 = np.mean(fT2)
MfT_s2 = np.mean(fT_s2)

#"händische" Fehlerrechung für erste Messreihe

#ich weiß nicht warum das so umständlich ist xD 

meanTl= 0
meanTr= 0
meanTp= 0
meanTm= 0
meanT= 0
meanTs = 0
for i in range(10):
    meanTl = meanTl + T_l[i]
meanTl = meanTl/10

for i in range(10):
    meanTr = meanTr + T_r[i]
meanTr = meanTr/10

for i in range(10):
    meanTp = meanTp + T_p[i]
meanTp = meanTp/10

for i in range(10):
    meanTm = meanTm + T_m[i]
meanTm = meanTm/10

for i in range(10):
    meanT = meanT + T[i]
meanT = meanT/10

for i in range(10):
    meanTs = meanTs + T_s[i]
meanTs = meanTs/10

meanTl2= 0
meanTr2= 0
meanTp2= 0
meanTm2= 0
meanT2= 0
meanTs2 = 0
for i in range(10):
    meanTl2 = meanTl2 + T_l2[i]
meanTl2 = meanTl2/10

for i in range(10):
    meanTr2 = meanTr2 + T_r2[i]
meanTr2 = meanTr2/10

for i in range(10):
    meanTp2 = meanTp2 + T_p2[i]
meanTp2 = meanTp2/10

for i in range(10):
    meanTm2 = meanTm2 + T_m2[i]
meanTm2 = meanTm2/10

for i in range(10):
    meanT2 = meanT2 + T2[i]
meanT2 = meanT2/10

for i in range(10):
    meanTs2 = meanTs2 + T_s2[i]
meanTs2 = meanTs2/10

#print(meanTl, meanTr, meanTp, meanTm, meanT, meanTs)
#print(meanTl2, meanTr2, meanTp2, meanTm2, meanT2, meanTs2)

#print(MfT_l, MfT_r, MfT_p,MfT_m,MfT,MfT_s)
#print(MfT_l2, MfT_r2, MfT_p2,MfT_m2,MfT2,MfT_s2)