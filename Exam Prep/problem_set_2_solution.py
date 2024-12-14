import numpy as np
import matplotlib.pyplot as plt

N, M = 8, 9
K = 100
T = np.array([0,5/24,11/24,17/24,23/24,29/24,35/24,41/24,47/24])
C = np.zeros([N,M])
L_3M, L_6M = 0.052, 0.049
R_1, R_2, R_3 = 0.051, 0.044, 0.049
pi = np.array([0.79492002,-1.02540877,2.05066409,103.02163487,101.8015268,104.48120266,101.10990798,103.67216735])

# Problem b)
C[0,:] = [0,-K*(0.25*L_3M+1),0.5*K*R_1,0,0.5*K*R_1+K,0,0,0,0]
C[1,:] = [0,0,K*(0.5*L_6M+1),0,-K*R_2,0,0,0,-K*R_2-K]
C[2,:] = [0,-K*(0.25*L_3M+1),0,0,K*R_3,0,0,0,K*R_3+K]
C[3,:] = [0,0.25*K*0.07,0.25*K*0.07,0.25*K*0.07,0.25*K*0.07+K,0,0,0,0]
C[4,:] = [0,0,0.5*K*0.05,0,0.5*K*0.05,0,0.5*K*0.05+K,0,0]
C[5,:] = [0,0,0,0,K*0.06,0,0,0,K*0.06+K]
C[6,:] = [0,0.25*K*0.045,0.25*K*0.045,0.25*K*0.045,0.25*K*0.045,0.25*K*0.045,0.25*K*0.045+K,0,0]
C[7,:] = [0,0.25*K*0.055,0.25*K*0.055,0.25*K*0.055,0.25*K*0.055,0.25*K*0.055,0.25*K*0.055,0.25*K*0.055,0.25*K*0.055+K]
print(f"Rank of matrix: {np.linalg.matrix_rank(C)}")

# Problem c)
p = np.ones([M])
y = pi - C[:,0]
x = C[:,1:]
p[1:] = np.linalg.solve(x,y)
print(f"ZCB prices: {p}")

# Problem d)
r, f_3M = np.zeros([M]), np.zeros([M])
for i in range(1,M):
    r[i] = -np.log(p[i])/T[i]
    f_3M[i] = -(np.log(p[i])-np.log(p[i-1]))/(T[i]-T[i-1])
print(f"ZCB spot rates: {r}")
print(f"3M forward rates: {f_3M}")

# problem e)
r_new, p_new = np.zeros([M]), np.ones([M])
for i in range(1,M):
    r_new[i] = r[i] - 0.001
    p_new[i] = np.exp(-T[i]*r_new[i])
pi_new = np.matmul(C,p_new)
print(f"New spot rates: {r_new}")
print(f"New ZCB prices: {p_new}")
print(f"New bond prices: {pi_new}")

# Problem f)
R_swap = (K*(0.25*L_3M+1)*p[1] - K*p[4])/(K*0.5*p[2]+K*0.5*p[4] )
print(f"R_swap if issued today: {R_swap}")

# Problem g)
R_trader = 0.052
C_trader = [0,0,0.5*K*R_trader,0,0.5*K*R_trader,0,0.5*K*R_trader,0,0.5*K*R_trader+K]
pi_trader = np.matmul(C_trader,p)
print(f"C_trader: {C_trader}, pi_trader: {pi_trader}")

# Problem h)
y = np.transpose(C_trader[1:])
x = np.transpose(C[:,1:])
h_rep = np.linalg.solve(x,y)
pi_rep = np.matmul(h_rep,pi)
print(f"arbitrage portfolio: {h_rep}")
print(f"arbitrage portfolio cashflow check: {np.matmul(h_rep,C)-C_trader}")
print(f"pi_rep: {pi_rep}")

C_zcb_3M = np.zeros([M])
C_zcb_3M[1] = 1
y = np.transpose(C_zcb_3M[1:])
h_zcb_3M = np.linalg.solve(x,y)
C_zcb_3M = np.matmul(np.transpose(h_zcb_3M),C)
pi_zcb_3M = np.matmul(h_zcb_3M,pi)
print(f"Portfolio replicating zcb_3M: {h_zcb_3M}")
print(f"Cashflow h_zcb_3M: {C_zcb_3M}")
print(f"Price h_zcb_3M: {pi_zcb_3M}")

fig = plt.figure(constrained_layout=False, dpi = 300, figsize = (5,3))   #
fig.suptitle(f"ZCB prices, spot rates and forward rates - question d)", fontsize = 9)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])

xtick_labels = [round(T[i],2) for i in range(0,M)]
ax.set_xticks(T)
ax.set_xticklabels(xtick_labels,fontsize = 6)
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

plt.show()
