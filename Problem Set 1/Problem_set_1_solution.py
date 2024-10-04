import numpy as np
import matplotlib.pyplot as plt

N, M = 6, 9
K = 100
T = np.array([0,0.25,0.5,0.75,1,1.25,1.5,1.75,2])
L_3M, L_6M = 0.01472717, 0.01893706
pi = np.array([102.33689177,104.80430234,105.1615306,105.6581905,104.02899992,101.82604116])
pi_new = np.array([101.98667646,102.96333877,102.68122237,104.17312216,102.76231402,98.8174065])
p_0_2M_new, p_0_5M_new = 0.99699147, 0.99088748
pi_swap_new = 0.02515099

# Problem a)
C = np.zeros([N,M])
C[0,:] = [0.04*K*0.25,0.04*K*0.25,0.04*K*0.25,0.04*K*0.25,0.04*K*0.25+K,0,0,0,0]
C[1,:] = [0.05*K*0.5,0,0.05*K*0.5,0,0.05*K*0.5+K,0,0,0,0]
C[2,:] = [0.05*K*0.5,0,0.05*K*0.5,0,0.05*K*0.5,0,0.05*K*0.5+K,0,0]
C[3,:] = [0.06*K*0.25,0.06*K*0.25,0.06*K*0.25,0.06*K*0.25,0.06*K*0.25,0.06*K*0.25,0.06*K*0.25+K,0,0]
C[4,:] = [0.05*K*0.25,0.05*K*0.25,0.05*K*0.25,0.05*K*0.25,0.05*K*0.25,0.05*K*0.25,0.05*K*0.25,0.05*K*0.25,0.05*K*0.25+K]
C[5,:] = [0.03*K*1,0,0,0,0.03*K*1,0,0,0,0.03*K*1+K]

# Problem b)
p, r = np.ones([M]), np.zeros([M])
p[1], p[2] = 1/(1+L_3M*T[1]), 1/(1+L_6M*T[2])
y = pi-p[0]*C[:,0]-p[1]*C[:,1]-p[2]*C[:,2]
x = C[:,3:]
p[3:M] = np.linalg.solve(x,y)
for i in range(1,M):
    r[i] = -np.log(p[i])/T[i]

# Problem c)
f_3M = np.zeros([M])
f_3M[0] = r[0]
for i in range(1,M):
    f_3M[i] = -(np.log(p[i])-np.log(p[i-1]))/(T[i]-T[i-1])

# Problem d)
pi_float = K*p[0]

# Problem e)
R = (1-p[8])/(0.5*(p[2]+p[4]+p[6]+p[8]))

T_new = np.zeros([M])
T_new[1:] = T[1:] - 1/12
C_new = np.zeros([N,M])
C_new[:,1:] = C[:,1:]

# problem g)
C_swap = np.array([0,-0.25*K*L_3M-K,0.5*R*K,0,0.5*R*K,0,0.5*R*K,0,0.5*R*K+K])

# problem h)
p_new = np.ones([M])
p_new[0], p_new[1], p_new[2] = 1, p_0_2M_new, p_0_5M_new
C_new = np.vstack([C_new,C_swap])
y = np.hstack([pi_new,pi_swap_new]) - p_new[1]*C_new[:,1] - p_new[2]*C_new[:,2]
x = C_new[:,3:]
p_new[3:] = np.linalg.solve(np.matmul(np.transpose(x),x),np.matmul(np.transpose(x),y))
r_new, f_3M_new = np.zeros([M]), np.zeros([M])
f_3M_new[0] = np.nan
for i in range(1,M):
    r_new[i] = -np.log(p_new[i])/T_new[i]
    f_3M_new[i] = -(np.log(p_new[i])-np.log(p_new[i-1]))/(T_new[i]-T_new[i-1])

# Plot for quextions a) - e)
fig = plt.figure(constrained_layout=False, dpi = 300, figsize = (5,3))   #
fig.suptitle(f"ZCB prices, spot rates and forward rates - questions a) to f)", fontsize = 9)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])

ax.set_xticks(T)
ax.set_xticklabels(T,fontsize = 6)
ax.set_xlim([T[0]-0.05,T[-1]+0.05])
plt.xlabel(f"Maturity",fontsize = 6)
ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1],fontsize = 6)
ax.set_ylim([0,1.05])
ax.set_ylabel(f"ZCB prices",fontsize = 6)
ax.scatter(T, p, s = 2, color = 'black', marker = ".")

ax1 = ax.twinx()
ax1.set_yticks([0,0.01,0.02,0.03,0.04,0.05])
ax1.set_yticklabels([0,0.01,0.02,0.03,0.04,0.05,0.05],fontsize = 6)
ax1.set_ylim([0,0.0525])
ax1.scatter(T, r, s = 2, color = 'red', marker = ".")
ax1.scatter(T, f_3M, s = 2, color = 'blue', marker = ".")
ax1.plot(T,R*np.ones([M]), color='green', linewidth = 1, marker = "")

# Plot for quextions g) - h)
fig2 = plt.figure(constrained_layout=False, dpi = 300, figsize = (5,3))   #
fig2.suptitle(f"ZCB prices, spot rates and forward rates - questions g) and h)", fontsize = 9)
gs = fig2.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig2.add_subplot(gs[0,0])

xtick_labels = [round(T_new[i],2) for i in range(0,M)]
ax.set_xticks(T_new)
ax.set_xticklabels(xtick_labels,fontsize = 6)
ax.set_xlim([T_new[0]-0.05,T_new[-1]+0.05])
plt.xlabel(f"Maturity",fontsize = 6)
ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1],fontsize = 6)
ax.set_ylim([0,1.05])
ax.set_ylabel(f"ZCB prices",fontsize = 6)
ax.scatter(T_new, p_new, s = 2, color = 'black', marker = ".")

ax1 = ax.twinx()
ax1.set_yticks([0,0.01,0.02,0.03,0.04,0.05])
ax1.set_yticklabels([0,0.01,0.02,0.03,0.04,0.05,0.05],fontsize = 6)
ax1.set_ylim([0,0.0525])
ax1.scatter(T_new, r_new, s = 2, color = 'red', marker = ".")
ax1.scatter(T_new, f_3M_new, s = 2, color = 'blue', marker = ".")

plt.show()
