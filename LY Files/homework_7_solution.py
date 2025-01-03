import numpy as np
import fixed_income_derivatives_E2023 as fid
import matplotlib.pyplot as plt

# Problem 1
delta = 0.25
M = 9
p = np.array([1,0.99228605,0.98417493,0.97571048,0.96693366,0.95788267,0.94859299,0.93909755,0.92942683])
T = np.array([m*delta for m in range(0,M)])

R = 0.045
sigma_market = np.array([np.nan,np.nan,0.15,0.19,0.23,0.27,0.3,0.33,0.36])

# a)
r, f_3m = np.zeros([M]), np.zeros([M])
for i in range(1,M):
    r[i] = - np.log(p[i])/T[i]
    f_3m[i] = -(np.log(p[i])-np.log(p[i-1]))/(T[i]-T[i-1])

# c)
alpha, L, price_caplet = np.zeros([M]), np.zeros([M]), np.zeros([M])
for i in range(1,M):
    alpha[i] = T[i]-T[i-1]
    L[i] = (1/alpha[i])*(p[i-1] - p[i])/p[i]
    if i > 1:
        price_caplet[i] = fid.black_caplet_price(sigma_market[i],T[i],R,alpha[i],p[i],L[i],type = 'call')
price_cap = sum(price_caplet[2:])
print(f"cap price: {price_cap} GBP.")

# d)
accrual_factor = 0
for i in range(1,M):
    accrual_factor += alpha[i]*p[i]
premium = price_cap/accrual_factor
print(f"premium: {premium} GBP, premium: {int(premium*10000)} bps.")

# e)
R = (1-p[M-1])/accrual_factor
print(f"Par swap rate: {R}.")

fig = plt.figure(constrained_layout=False, dpi = 300, figsize = (5,3))   #
fig.suptitle(f"Spot rates and 3M forward rates", fontsize = 9)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])

xticks = [0,0.5,1,1.5,2]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 6)
ax.set_xlim([xticks[0]-0.02,xticks[-1]+0.02])
plt.xlabel(f"Maturity",fontsize = 6)
ax.set_yticks([0,0.01,0.02,0.03,0.04,0.05])
ax.set_yticklabels([0,0.01,0.02,0.03,0.04,0.05],fontsize = 6)
ax.set_ylim([0,0.0525])
# ax.set_ylabel(f"",fontsize = 6)
p1 = ax.scatter(T[1:], r[1:], s = 1, color = 'black', marker = ".",label="Spot rates")
p2 = ax.scatter(T[1:], f_3m[1:], s = 1, color = 'red', marker = ".",label="3M forward rates")
p3 = ax.scatter(T[1:], L[1:], s = 1, color = 'green', marker = ".",label="3M forward LIBOR rates")

plots = [p1,p2,p3]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="upper right",fontsize = 6)

plt.show()

# Problem 2
sigma = 0.25
precision = 1e-10
T, R, alpha, p, L = 2, 0.04, 0.5, 0.92, 0.04
price_caplet = fid.black_caplet_price(sigma,T,R,alpha,p,L,type = 'call')
iv_caplet = fid.black_caplet_iv(price_caplet,T,R,alpha,p,L, iv0 = 0.2, max_iter = 200, prec = precision)

# Problem 3
M, delta = 9, 1/2
T = np.array([i*delta for i in range(0,M)])
C = np.array([0,0,0.00062138,0.00193406,0.00329997,0.00462232,0.00588119,0.00707032,0.00818548])
p = np.array([1,0.98530023,0.96939649,0.95255338,0.93499966,0.91693156,0.89851614,0.87989458,0.86118526])
R = 0.035

# a)
R_spot = fid.zcb_to_spot_rates(T,p)
f = fid.zcb_to_forward_rates(T,p,horizon = 1)
L = fid.zcb_to_forward_LIBOR_rates(T,p,horizon = 1)

# b)
iv = np.zeros([M])
for i in range(2,M):                        # sigma,T,R,alpha,p,L,type = "call"
    iv[i] = fid.black_caplet_iv(C[i],T[i],R,T[i]-T[i-1],p[i],L[i], iv0 = 0.2, max_iter = 400, prec = 1.0e-15)
print(f"Term Structure of Implied Volatility: {iv}")

# c) and d)
N = M - 2
y = iv[2:]**2
A = 0.5*np.tri(N)
for i in range(2,M):
    A[i-2,:] = A[i-2,:]/T[i-1]
beta = (np.linalg.solve(A,y))**0.5
print(f"beta vector: {beta}")

fig = plt.figure(constrained_layout=False, dpi = 300, figsize = (5,3))   #
fig.suptitle(f"Spot rates, 3M forward rates and LIBOR rates", fontsize = 9)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])

xticks = [0,1,2,3,4]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 6)
ax.set_xlim([xticks[0]-0.02,xticks[-1]+0.02])
plt.xlabel(f"Maturity",fontsize = 6)
ax.set_yticks([0,0.01,0.02,0.03,0.04,0.05])
ax.set_yticklabels([0,0.01,0.02,0.03,0.04,0.05],fontsize = 6)
ax.set_ylim([0,0.0525])
# ax.set_ylabel(f"",fontsize = 6)
p1 = ax.scatter(T[1:], R_spot[1:], s = 1, color = 'black', marker = ".",label="Spot rates")
p2 = ax.scatter(T[1:], f[1:], s = 1, color = 'red', marker = ".",label="3M forward rates")
p3 = ax.scatter(T[1:], L[1:], s = 1, color = 'green', marker = ".",label="3M forward LIBOR rates")

plots = [p1,p2,p3]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="lower right",fontsize = 6)

plt.show()
