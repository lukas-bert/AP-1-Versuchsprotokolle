import matplotlib.pyplot as plt
import numpy as np
 
f, U, V, a, b= np.genfromtxt("data_c_d.txt", unpack = True)

phi = a/b *2 *np.pi
np.savetxt('Phasenverschiebung.txt', phi, fmt='%.2f', header="Phi/rad") # zum Rechnen mit phi solllte fmt='%.18e' gesetzt werden / gelöscht werden
