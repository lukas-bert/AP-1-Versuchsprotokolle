import matplotlib.pyplot as plt
import numpy as np

I, B = np.genfromtxt("content/dataHysterese.txt", unpack = True)

plt.subplot(1, 2, 1)
plt.plot(I, B, 'r.', label='Messwerte')
plt.xlabel(r'$\alpha \mathbin{/} \unit{\ohm}$')
plt.ylabel(r'$y \mathbin{/} \unit{\micro\joule}$')
plt.legend(loc='best')
plt.savefig('build/plotHysterese.pdf')
