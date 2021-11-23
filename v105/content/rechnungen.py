# Zschwischenrechnungen zur Messwertverarbeitung
import numpy as np

# Mitteln der Schwingungsdauer T
I, T10_1, T10_2 = np.genfromtxt("Messung2.txt", unpack = True)

T10 = (T10_1 + T10_2)/2
T = T10/10

np.savetxt("Messung2_T.txt",np.transpose([I,T]) , fmt = '%.2f', header="# I/A, T/s")