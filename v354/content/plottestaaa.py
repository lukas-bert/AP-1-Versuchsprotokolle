import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

def ExpFit(x, m, tt):
    return m * np.exp(-tt * x)
t_0, U_0 = np.genfromtxt("data_a.txt", unpack = True)
t = t_0 * 0.05
errt = 0.025
errU_0 = 0.05
p0 = (0, 3.4)
params = scipy.optimize.curve_fit(ExpFit, t, U_0, p0)
m, tt = params
#fit = np.polyfit(t, (-1)*np.log(U_0),1)
#x = np.linspace(0,1,100)
#y = np.exp(fit[0]) + (np.exp(fit[1]))**x
#print(fit)

#sampleRate = 20_000 # Hz
#tauSec = (1 / tt) / sampleRate

# determine quality of the fit
#squaredDiffs = np.square(U_0 - monoExp(t, m, tt, b))
#squaredDiffsFromMean = np.square(U_0 - np.mean(U_0))
#rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
#print(f"R² = {rSquared}")
#(noch evtl zur berechnung nutzen) print(m, tt, b)

#Plots
plt.plot(t, ExpFit(t, m, tt), 'r--')
plt.errorbar(t, U_0,xerr = errt, yerr = errU_0, fmt='o')
#plt.plot(x,y)
#Aussehen
plt.yscale('log')
plt.xlabel("$t /$ " "$\mathrm{\mu}$" r'$\mathrm{s}$')
plt.ylabel("$U_0 /$ "  r'$\mathrm{V}$')
plt.grid()
plt.savefig('plottestaaa.pdf')