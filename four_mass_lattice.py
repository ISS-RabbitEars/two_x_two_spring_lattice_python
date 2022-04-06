import numpy as np
import sympy as sp
from sympy.physics.vector import dynamicsymbols
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import animation


def integrate(ic, ti, p):
	m, k, req, xp, yp = p
	ic_list = ic

	sub = {}
	for i in range(12): 
		sub[K[i]] = k[i]
		sub[Req[i]] = req[i]
		if i < 8:
			sub[Xp[i]] = xp[i]
			sub[Yp[i]] = yp[i]
		if i < 4:
			sub[M[i]] = m[i]
			sub[R[i]] = ic_list[4 * i]
			sub[Rdot[i]] = ic_list[4 * i + 1]
			sub[THETA[i]] = ic_list[4 * i + 2]
			sub[THETAdot[i]] = ic_list[4 * i + 3]

	diff_eqs = []
	for i in range(4):
		diff_eqs.append(ic_list[4 * i + 1])
		diff_eqs.append(A[i].subs(sub))
		diff_eqs.append(ic_list[4 * i + 3])
		diff_eqs.append(ALPHA[i].subs(sub))

	print(ti)

	return diff_eqs


t = sp.Symbol('t')
M = sp.symbols('M0:4')
K = sp.symbols('K0:12')
Req = sp.symbols('Req0:12')
Xp = sp.symbols('Xp0:8')
Yp = sp.symbols('Yp0:8')
R = dynamicsymbols('R0:4')
THETA = dynamicsymbols('THETA0:4')

Rdot = np.asarray([i.diff(t, 1) for i in R])
Rddot = np.asarray([i.diff(t, 2) for i in R])
THETAdot = np.asarray([i.diff(t, 1) for i in THETA])
THETAddot = np.asarray([i.diff(t, 2) for i in THETA])

X = [R[i] * sp.cos(THETA[i]) for i in range(4)]
Y = [R[i] * sp.sin(THETA[i]) for i in range(4)]

Xdot = np.asarray([i.diff(t, 1) for i in X])
Ydot = np.asarray([i.diff(t, 1) for i in Y])

T = sp.simplify(sp.Rational(1, 2) * sum(M * (Xdot**2 + Ydot**2)))

dR1 = np.asarray([sp.sqrt((R[i] * sp.cos(THETA[i]) - Xp[2 * i + j])**2 + (R[i] * sp.sin(THETA[i]) - Yp[2 * i + j])**2) for j in range(2) for i in range(4)])
dR2 = np.asarray([sp.sqrt((X[(i+1)%4] - X[i])**2 + (Y[(i+1)%4] - Y[i])**2) for i in range(4)])
dR = np.append(dR1,dR2)

V = sp.simplify(sp.Rational(1, 2) * sum(K * (dR - Req)**2))

L = T - V

dLdR = np.asarray([L.diff(i, 1) for i in R])
dLdRdot = np.asarray([L.diff(i, 1) for i in Rdot])
ddtdLdRdot = np.asarray([i.diff(t, 1) for i in dLdRdot])
dLR = ddtdLdRdot - dLdR

dLdTHETA = np.asarray([L.diff(i, 1) for i in THETA])
dLdTHETAdot = np.asarray([L.diff(i, 1) for i in THETAdot])
ddtdLdTHETAdot = np.asarray([i.diff(t, 1) for i in dLdTHETAdot])
dLTHETA = ddtdLdTHETAdot - dLdTHETA

sol = sp.solve(np.append(dLR,dLTHETA).tolist(), np.append(Rddot,THETAddot).tolist())

A = [sp.simplify(sol[i]) for i in Rddot]
ALPHA = [sp.simplify(sol[i]) for i in THETAddot]

#-----------------------------------------------------------------

m = np.asarray([1, 1, 1, 1])
k = np.asarray([25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25 ,25])
req= np.asarray([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
xp = np.asarray([10, 5, -5, -10, -10, -5, 5, 10])
yp = np.asarray([5, 10, 10, 5, -5, -10, -10, -5])
ro = np.asarray([7.5, 3.5, 7.5, 3.5])
vo = np.asarray([0, 0, 0, 0])
thetao = np.asarray([45, 135, 225, 315]) * np.pi/180
omegao = np.asarray([0, 0, 0, 0]) * np.pi/180 
tf = 30

p = m, k, req, xp, yp

ic = []
for i in range(4):
	ic.append(ro[i])
	ic.append(vo[i])
	ic.append(thetao[i])
	ic.append(omegao[i])

nfps = 30
nframes = tf * nfps
ta = np.linspace(0, tf, nframes)

rth = odeint(integrate, ic, ta, args = (p,))

x = np.asarray([[X[i].subs({R[i]:rth[j,4 * i], THETA[i]:rth[j,4 * i + 2]}) for j in range(nframes)] for i in range(4)],dtype=float)
y = np.asarray([[Y[i].subs({R[i]:rth[j,4 * i], THETA[i]:rth[j,4 * i + 2]}) for j in range(nframes)] for i in range(4)],dtype=float)

ke = np.zeros(nframes)
pe = np.zeros(nframes)
for j in range(nframes):
	sub = {}
	for i in range(12):
		sub[K[i]] = k[i]
		sub[Req[i]] = req[i]
		if i < 8:
			sub[Xp[i]] = xp[i]
			sub[Yp[i]] = yp[i]
		if i < 4:
			sub[M[i]] = m[i]
			sub[R[i]] = rth[j, 4 * i]
			sub[Rdot[i]] = rth[j, 4 * i + 1]
			sub[THETA[i]] = rth[j, 4 * i + 2]
			sub[THETAdot[i]] = rth[j, 4 * i + 3]
	ke[j] = T.subs(sub)
	pe[j] = V.subs(sub)
E = ke + pe

#----------------------------------------------------------

xmax = x.max() if x.max() > max(xp) else max(xp)
xmin = x.min() if x.min() < min(xp) else min(xp)
ymax = y.max() if y.max() > max(yp) else max(yp)
ymin = y.min() if y.min() < min(xp) else min(xp)

msf = 1/75
drs = np.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)
mr = msf * drs
mra = mr * m / max(m)
mras = np.asarray([mra[i]+mra[(i+1)%4] for i in range(4)])

xmax += 2*max(mra)
xmin -= 2*max(mra)
ymax += 2*max(mra)
ymin -= 2*max(mra)

dr1 = np.asarray([np.sqrt((x[(i+1)%4] - x[i])**2 + (y[(i+1)%4] - y[i])**2) for i in range(4)])
dr2 = np.asarray([np.sqrt((xp[2*i+j] - x[i])**2 + (yp[2*i+j] - y[i])**2) for j in range(2) for i in range(4)])
dr1max = np.asarray([max(dr1[i]) for i in range(4)])
dr2max = np.asarray([max(dr2[i]) for i in range(8)])
theta1 = np.asarray([np.arccos((y[(i+1)%4] - y[i])/dr1[i]) for i in range(4)])
theta2 = np.asarray([np.arccos((yp[i] - y[i//2])/dr2[i]) for i in range(8)])
nl1 = np.asarray([np.ceil(dr1max[i]/mras[i]) for i in range(4)],dtype=int)
nl2 = np.asarray([np.ceil(dr2max[i]/(2*mra[i//2])) for i in range(8)],dtype=int)
l1 = np.asarray([(dr1[i] - mras[i])/nl1[i] for i in range(4)])
l2 = np.asarray([(dr2[i] - mra[i//2])/nl2[i] for i in range(8)])
h1 = np.asarray([np.sqrt((mras[i]/2)**2 - (0.5 * l1[i])**2) for i in range(4)])
h2 = np.asarray([np.sqrt(mra[i//2]**2 - (0.5 * l2[i])**2) for i in range(8)])
flip1a = np.zeros((4,nframes))
flip1b = np.zeros((4,nframes))
flip1c = np.zeros((4,nframes))
for i in range(4):
	flip1a[i] = np.asarray([-1 if x[i][j]>x[(i+1)%4][j] and y[i][j]<y[(i+1)%4][j] else 1 for j in range(nframes)])
	flip1b[i] = np.asarray([-1 if x[i][j]<x[(i+1)%4][j] and y[i][j]>y[(i+1)%4][j] else 1 for j in range(nframes)])
	flip1c[i] = np.asarray([-1 if x[i][j]<x[(i+1)%4][j] else 1 for j in range(nframes)])
flip2a = np.zeros((8,nframes))
flip2b = np.zeros((8,nframes))
flip2c = np.zeros((8,nframes))
for i in range(8):
	flip2a[i] = np.asarray([-1 if x[i//2][j]>xp[i] and y[i//2][j]<yp[i] else 1 for j in range(nframes)])	
	flip2b[i] = np.asarray([-1 if x[i//2][j]<xp[i] and y[i//2][j]>yp[i] else 1 for j in range(nframes)])
	flip2c[i] = np.asarray([-1 if x[i//2][j]<xp[i] else 1 for j in range(nframes)])
xlo1 = np.zeros((4,nframes))
ylo1 = np.zeros((4,nframes))
for i in range(4):
	xlo1[i] = x[i] + np.sign((y[(i+1)%4] - y[i]) * flip1a[i] * flip1b[i]) * mra[i] * np.sin(theta1[i])
	ylo1[i] = y[i] + mra[i] * np.cos(theta1[i])
xlo2 = np.zeros((8,nframes))
ylo2 = np.zeros((8,nframes))
for i in range(8):
        xlo2[i] = x[i//2] + np.sign((yp[i] - y[i//2]) * flip2a[i] * flip2b[i]) * mra[i//2] * np.sin(theta2[i])
        ylo2[i] = y[i//2] + mra[i//2] * np.cos(theta2[i])
xl1 = np.zeros((4,max(nl1),nframes))
yl1 = np.zeros((4,max(nl1),nframes))
for i in range(4):
	for j in range(nl1[i]):
		xl1[i][j] = xlo1[i] + np.sign((y[(i+1)%4]-y[i])*flip1a[i]*flip1b[i]) * (0.5 + j) * l1[i] * np.sin(theta1[i]) - np.sign((y[(i+1)%4]-y[i])*flip1a[i]*flip1b[i]) * flip1c[i] * (-1)**j * h1[i] * np.sin(np.pi/2 - theta1[i])
		yl1[i][j] = ylo1[i] + (0.5 + j) * l1[i] * np.cos(theta1[i]) + flip1c[i] * (-1)**j * h1[i] * np.cos(np.pi/2 - theta1[i])
xl2 = np.zeros((8,max(nl2),nframes))
yl2 = np.zeros((8,max(nl2),nframes))
for i in range(8):
	for j in range(nl2[i]):
		xl2[i][j] = xlo2[i] + np.sign((yp[i]-y[i//2])*flip2a[i]*flip2b[i]) * (0.5 + j) * l2[i] * np.sin(theta2[i]) - np.sign((yp[i]-y[i//2])*flip2a[i]*flip2b[i]) * flip2c[i] * (-1)**j * h2[i] * np.sin(np.pi/2 - theta2[i])
		yl2[i][j] = ylo2[i] + (0.5 + j) * l2[i] * np.cos(theta2[i]) + flip2c[i] * (-1)**j * h2[i] * np.cos(np.pi/2 - theta2[i])
xlf1 = np.zeros((4,nframes))
ylf1 = np.zeros((4,nframes))
for i in range(4):
	xlf1[i] = x[(i+1)%4] - mra[(i+1)%4] * np.sign((y[(i+1)%4]-y[i])*flip1a[i]*flip1b[i]) * np.sin(theta1[i])
	ylf1[i] = y[(i+1)%4] - mra[(i+1)%4] * np.cos(theta1[i])

fig, a=plt.subplots()

def run(frame):
	plt.clf()
	plt.subplot(211)
	for i in range(4):
		circle=plt.Circle((x[i][frame],y[i][frame]),radius=mra[i],fc='xkcd:red')
		plt.gca().add_patch(circle)
	for i in range(8):
		circle=plt.Circle((xp[i],yp[i]),radius=max(mra)/2,fc='xkcd:cerulean')
		plt.gca().add_patch(circle)
	for i in range(4):
		plt.plot([xlo1[i][frame],xl1[i][0][frame]],[ylo1[i][frame],yl1[i][0][frame]],'xkcd:cerulean')
		plt.plot([xl1[i][nl1[i]-1][frame],xlf1[i][frame]],[yl1[i][nl1[i]-1][frame],ylf1[i][frame]],'xkcd:cerulean')
		for j in range(nl1[i]-1):
			plt.plot([xl1[i][j][frame],xl1[i][j+1][frame]],[yl1[i][j][frame],yl1[i][j+1][frame]],'xkcd:cerulean')
	for i in range(8):
		plt.plot([xlo2[i][frame],xl2[i][0][frame]],[ylo2[i][frame],yl2[i][0][frame]],'xkcd:cerulean')
		plt.plot([xl2[i][nl2[i]-1][frame],xp[i]],[yl2[i][nl2[i]-1][frame],yp[i]],'xkcd:cerulean')
		for j in range(nl2[i]-1):
			plt.plot([xl2[i][j][frame],xl2[i][j+1][frame]],[yl2[i][j][frame],yl2[i][j+1][frame]],'xkcd:cerulean')
	plt.title("Mass-Spring Lattice")
	ax=plt.gca()
	ax.set_aspect(1)
	plt.xlim([float(xmin),float(xmax)])
	plt.ylim([float(ymin),float(ymax)])
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.set_facecolor('xkcd:black')
	plt.subplot(212)
	plt.plot(ta[0:frame],ke[0:frame],'xkcd:red',lw=1.0)
	plt.plot(ta[0:frame],pe[0:frame],'xkcd:cerulean',lw=1.0)
	plt.plot(ta[0:frame],E[0:frame],'xkcd:bright green',lw=1.5)
	plt.xlim([0,tf])
	plt.title("Energy")
	ax=plt.gca()
	ax.legend(['T','V','E'],labelcolor='w',frameon=False)
	ax.set_facecolor('xkcd:black')

ani=animation.FuncAnimation(fig,run,frames=nframes)
writervideo = animation.FFMpegWriter(fps=nfps)
ani.save('four_mass_lattice.mp4', writer=writervideo)
plt.show()







