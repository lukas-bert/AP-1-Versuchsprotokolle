import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op

c_L = 1800
c_P = 2700
nu_0 = 2*10**6
theta = [np.pi/6, np.pi/12, np.pi/4]
alpha = (np.pi/2)-np.arcsin(np.sin(theta)*(c_L/c_P))

#Messaufgabe 1

# kleines Rohr
rpm, v_diff30, v_diff15, v_diff45 = np.genfromtxt("content/data/Mess1_klein.txt", unpack = True)

v_fluss30 = v_diff30*c_L/(2*nu_0*np.cos(alpha[0]))
v_fluss15 = v_diff15*c_L/(2*nu_0*np.cos(alpha[1]))
v_fluss45 = v_diff45*c_L/(2*nu_0*np.cos(alpha[2]))

#plt.subplot(1,3,1)
plt.xlabel(r'$v_\text{Fluss} \mathbin{/} \unit{\metre\per\second}$')
plt.ylabel(r'$\frac{\symup{\Delta}\nu}{\text{cos}\,\alpha}\mathbin{/}\unit{\hertz}$')
#plt.title(r"$\theta = 30\unit{\degree}$")
plt.grid()
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.plot(v_fluss30,v_diff30/np.cos(alpha[0]), marker = "x" , color = "firebrick", linewidth = 0, label = r'$\theta = \qty{30}{\degree}$') 

#plt.subplot(1,3,2)
#plt.xlabel(r'$v_\text{Fluss} \mathbin{/} \unit{\metre\per\second}$')
#plt.title(r"$\theta = 15\unit{\degree}$")
#plt.grid()
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.plot(v_fluss15,v_diff15/np.cos(alpha[1]), marker = "+", markersize = 7 , color = "steelblue", linewidth = 0, label = r'$\theta = \qty{15}{\degree}$')

#plt.subplot(1,3,3)
#plt.xlabel(r'$v_\text{Fluss} \mathbin{/} \unit{\metre\per\second}$')
#plt.title(r"$\theta = 45\unit{\degree}$")
#plt.grid()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.plot(v_fluss45,v_diff45/np.cos(alpha[2]), marker = "3", markersize = 8 , color = "darkorchid", linewidth = 0,  label = r'$\theta = \qty{45}{\degree}$') 

plt.legend()
plt.savefig('build/plot1_1.pdf')
plt.close()

# mittleres Rohr
rpm, v_diff30, v_diff15, v_diff45 = np.genfromtxt("content/data/Mess1_mittel.txt", unpack = True)

v_fluss30 = v_diff30*c_L/(2*nu_0*np.cos(alpha[0]))
v_fluss15 = v_diff15*c_L/(2*nu_0*np.cos(alpha[1]))
v_fluss45 = v_diff45*c_L/(2*nu_0*np.cos(alpha[2]))

#plt.subplot(1,3,1)
plt.xlabel(r'$v_\text{Fluss} \mathbin{/} \unit{\metre\per\second}$')
plt.ylabel(r'$\frac{\symup{\Delta}\nu}{\text{cos}\,\alpha}\mathbin{/}\unit{\hertz}$')
#plt.title(r"$\theta = 30\unit{\degree}$")
plt.grid()
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.plot(v_fluss30,v_diff30/np.cos(alpha[0]), marker = "x" , color = "firebrick", linewidth = 0, label = r'$\theta = \qty{30}{\degree}$') 

#plt.subplot(1,3,2)
#plt.xlabel(r'$v_\text{Fluss} \mathbin{/} \unit{\metre\per\second}$')
#plt.title(r"$\theta = 15\unit{\degree}$")
#plt.grid()
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.plot(v_fluss15,v_diff15/np.cos(alpha[1]), marker = "+", markersize = 7 , color = "steelblue", linewidth = 0, label = r'$\theta = \qty{15}{\degree}$')

#plt.subplot(1,3,3)
#plt.xlabel(r'$v_\text{Fluss} \mathbin{/} \unit{\metre\per\second}$')
#plt.title(r"$\theta = 45\unit{\degree}$")
#plt.grid()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.plot(v_fluss45,v_diff45/np.cos(alpha[2]), marker = "3", markersize = 8 , color = "darkorchid", linewidth = 0,  label = r'$\theta = \qty{45}{\degree}$') 

plt.legend() 

plt.savefig('build/plot1_2.pdf')
plt.close()

# großes Rohr
rpm, v_diff30, v_diff15, v_diff45 = np.genfromtxt("content/data/Mess1_groß.txt", unpack = True)

v_fluss30 = v_diff30*c_L/(2*nu_0*np.cos(alpha[0]))
v_fluss15 = v_diff15*c_L/(2*nu_0*np.cos(alpha[1]))
v_fluss45 = v_diff45*c_L/(2*nu_0*np.cos(alpha[2]))

#plt.subplot(1,3,1)
plt.xlabel(r'$v_\text{Fluss} \mathbin{/} \unit{\metre\per\second}$')
plt.ylabel(r'$\frac{\symup{\Delta}\nu}{\text{cos}\,\alpha}\mathbin{/}\unit{\hertz}$')
#plt.title(r"$\theta = 30\unit{\degree}$")
plt.grid()
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.plot(v_fluss30,v_diff30/np.cos(alpha[0]), marker = "x" , color = "firebrick", linewidth = 0, label = r'$\theta = \qty{30}{\degree}$') 

#plt.subplot(1,3,2)
#plt.xlabel(r'$v_\text{Fluss} \mathbin{/} \unit{\metre\per\second}$')
#plt.title(r"$\theta = 15\unit{\degree}$")
#plt.grid()
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.plot(v_fluss15,v_diff15/np.cos(alpha[1]), marker = "+", markersize = 7 , color = "steelblue", linewidth = 0, label = r'$\theta = \qty{15}{\degree}$')

#plt.subplot(1,3,3)
#plt.xlabel(r'$v_\text{Fluss} \mathbin{/} \unit{\metre\per\second}$')
#plt.title(r"$\theta = 45\unit{\degree}$")
#plt.grid()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.plot(v_fluss45,v_diff45/np.cos(alpha[2]), marker = "3", markersize = 8 , color = "darkorchid", linewidth = 0,  label = r'$\theta = \qty{45}{\degree}$') 

plt.legend() 

plt.savefig('build/plot1_3.pdf')
plt.close()

# Messaufgabe 2

t, v_diff, v, I = np.genfromtxt("content/data/Mess2_45.txt", unpack = True)

#FIT

def f2(x,a,b,c):
    return a*x**2 + b*x + c

params, pcov = op.curve_fit(f2, t, I)
x = np.linspace(13,17.5,1000) 
plt.subplot(1,2,1)
plt.plot(t, I, marker = "x" ,color = "firebrick", linewidth = 0, label = "Messwerte")
plt.plot(x, f2(x, *params), color = "cornflowerblue", label = "Fit")

plt.legend()
plt.xlabel(r'Messtiefe in $\unit{\micro\second}$')
plt.ylabel(r'$I \mathbin{/} \unit{\kilo\volt\squared\per\second}$')
plt.xlim(13, 17.5)

plt.grid()
plt.title("Intensität")
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

params, pcov = op.curve_fit(f2, t, v)
x = np.linspace(13,17.5,1000) 

plt.subplot(1,2,2)
plt.plot(t, v, marker = "x" ,color = "firebrick", linewidth = 0)
plt.plot(x, f2(x, *params), color = "cornflowerblue", label = "Fit")


plt.xlabel(r'Messtiefe in $\unit{\micro\second}$')
plt.ylabel(r'$v \mathbin{/} \unit{\centi\metre\per\second}$')
plt.xlim(13, 17.5)

plt.grid()
plt.title("momentane Geschwindigkeit")
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

#plt.show()



plt.savefig('build/plot2_1.pdf')
plt.close()

t, v_diff, v, I = np.genfromtxt("content/data/Mess2_70.txt", unpack = True)

params, pcov = op.curve_fit(f2, t, I)
x = np.linspace(10,20,1000) 
plt.subplot(1,2,1)
plt.plot(t, I, marker = "x" ,color = "firebrick", linewidth = 0, label = "Messwerte")
plt.plot(x, f2(x, *params), color = "cornflowerblue", label = "Fit")

plt.legend()
plt.xlabel(r'Messtiefe in $\unit{\micro\second}$')
plt.ylabel(r'$I \mathbin{/} \unit{\kilo\volt\squared\per\second}$')
plt.xlim(11.5, 20)
plt.ylim(0)
plt.grid()
plt.title("Intensität")
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

params, pcov = op.curve_fit(f2, t[2:13], v[2:13])
x = np.linspace(10,20,1000) 

plt.subplot(1,2,2)
plt.plot(t[2:13], v[2:13], marker = "x" ,color = "firebrick", linewidth = 0)
plt.plot(x, f2(x, *params), color = "cornflowerblue", label = "Fit")


plt.xlabel(r'Messtiefe in $\unit{\micro\second}$')
plt.ylabel(r'$v \mathbin{/} \unit{\centi\metre\per\second}$')
plt.xlim(12.5, 18.5)
plt.ylim(0)
plt.grid()
plt.title("momentane Geschwindigkeit")
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
#plt.show()
plt.savefig('build/plot2_2.pdf')
plt.close()
