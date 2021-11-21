import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import uncertainties.unumpy as unp
from uncertainties import ufloat

R_f = ufloat(732,0.5)
L_f = ufloat(16.87*(10**(-3)),0.05*(10**(-3)))
C_f = ufloat(2.06*(10**(-9)),0.003*(10**(-9)))
f_f1 = ((R_f/(2*L_f))+unp.sqrt(((R_f**2)/4*(L_f**2))+(1/(L_f*C_f))))
f_f2 = ((-R_f/(2*L_f))+unp.sqrt(((R_f**2)/4*(L_f**2))+(1/(L_f*C_f))))
print(f_f1,f_f2)
F_f1 = f_f1/(2000*np.pi)
F_f2 = f_f2/(2000*np.pi)
print(F_f1,F_f2)