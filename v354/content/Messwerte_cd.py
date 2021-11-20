import matplotlib.pyplot as plt
import numpy as np
 
f, U_C, U_0, a, b = np.genfromtxt("data_c_d.txt", unpack = True)

phi = a/b *2 *np.pi
U_q = U_C/U_0

np.savetxt('Messwerte_cd.txt', np.transpose([phi, U_q]), fmt='%.2f', header="Phi/rad, U_C/U_0") # zum Rechnen mit phi solllte fmt='%.18e' gesetzt werden / gelöscht werden
