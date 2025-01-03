import numpy as np
import fixed_income_derivatives_E2024 as fid
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

T = np.array([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75,
5, 5.25, 5.5, 5.75, 6, 6.25, 6.5, 6.75, 7, 7.25, 7.5, 7.75, 8, 8.25, 8.5, 8.75, 9, 9.25, 9.5, 9.75, 10])
R = np.array([0.036, 0.03711, 0.03797, 0.03875, 0.03946, 0.0401, 0.04069, 0.04122, 0.04171, 0.04216, 0.04257, 0.04294, 0.04329,
0.0436, 0.0439, 0.04417, 0.04441, 0.04464, 0.04486, 0.04506, 0.04524, 0.04541, 0.04557, 0.04572, 0.04586, 0.04599, 0.04611,
0.04622, 0.04633, 0.04643, 0.04653, 0.04662, 0.0467, 0.04678, 0.04686, 0.04693, 0.047, 0.04706, 0.04712, 0.04718, 0.04724])

# Problem 3 - Fitting a Vasicek model to data
param_0 = 0.03, 0.5, 0.04, 0.03
result = minimize(fid.fit_vasicek_obj,param_0,method = 'nelder-mead',args = (R,T),options={'xatol': 1e-20,'disp': True})
print(f"Parameters from the fit: {result.x}. SSE of the fit: {result.fun}")
r0, a, b, sigma = result.x
p = fid.zcb_price_vasicek(r0,a,b,sigma,T)
f = fid.forward_rate_vasicek(r0,a,b,sigma,T)
T_swap = [1,2,3,4,5,6,7,8,9,10]
R_swap = np.zeros([10])
for i in range(0,10):
    R_swap[i] = fid.swap_rate_from_zcb_prices(0,0,T_swap[i],"annual",T,p)[0]
f_3m = fid.forward_rates_from_zcb_prices(T,p,horizon = 1)

# Problem 4 - Caplet and cap prices
strike = 0.05
alpha = 0.25
M = int(10/alpha) + 1
price_caplet = np.zeros([len(T)])
for i in range(2,len(T)):
    price_caplet[i] = (1 + (T[i]-T[i-1])*strike)*fid.euro_option_price_vasicek(1/(1 + (T[i]-T[i-1])*strike),T[i-1],T[i],p[i-1],p[i],a,sigma,type = "put")
price_cap = sum(price_caplet[2:])
S_swap = fid.accrual_factor_from_zcb_prices(0,0,T[-1],"annual",T,p)
premium_cap = alpha*(price_cap/S_swap)
print(f"Caplet prices: {10000*price_caplet}")
print(f"price_cap: {10000*price_cap}, premium_cap: {10000*premium_cap}")
price_caplet_down = fid.caplet_prices_vasicek(sigma-0.001,strike,a,T,p)
price_cap_down = sum(price_caplet_down[2:])
premium_cap_down = alpha*(price_cap_down/S_swap)
print(f"price_cap_down: {10000*price_cap_down}, premium_cap_down: {10000*premium_cap_down}")
price_caplet_up = fid.caplet_prices_vasicek(sigma+0.001,strike,a,T,p)
price_cap_up = sum(price_caplet_up[2:])
premium_cap_up = alpha*(price_cap_up/S_swap)
print(f"price_cap_up: {10000*price_cap_up}, premium_cap_up: {10000*premium_cap_up}")

# Problem 5 - Simulating the short rate in the Vasicek model
M_simul, T_simul = 1000, 10
size_ci = 0.95
r_simul = fid.simul_vasicek(r0,a,b,sigma,M_simul,T_simul,method = "euler")
t_simul = np.array([i*(T_simul/M_simul) for i in range(0,M_simul+1)])
lb, ub = fid.ci_vasicek(r0,a,b,sigma,t_simul,size_ci)
mu_sd, sigma_sd = b/a, sigma/(np.sqrt(2*a))
lb_sd = mu_sd - norm.ppf(size_ci + 0.5*(1-size_ci))*sigma_sd
ub_sd = mu_sd + norm.ppf(size_ci + 0.5*(1-size_ci))*sigma_sd

# Problem 6 - Computing the price of a 2Y8Y swaption
T_n, T_N = 2, 10
M_simul_swaption, N_simul, T_simul_swaption = 4000, 50000, T_n
chi, price_swaption_simul, price_swaption_plot = np.zeros([N_simul]), np.zeros([N_simul]), np.zeros([N_simul])
T_swaption = np.array([i*0.25 for i in range(0,33)])
for i in range(0,N_simul):
    r_simul_swaption = fid.simul_vasicek(r0,a,b,sigma,M,T_n,method = "exact",seed = None)
    p_swaption = fid.zcb_price_vasicek(r_simul_swaption[-1],a,b,sigma,T_swaption)
    R_swaption, S_swaption = fid.swap_rate_from_zcb_prices(0,0,10-2,"annual",T_swaption,p_swaption,float_freq = "quarterly")
    chi[i] = max(R_swaption - strike,0)*S_swaption
    price_swaption_simul[i] = np.exp(-(T_simul_swaption/M_simul_swaption)*sum(r_simul_swaption))*chi[i]
    price_swaption_plot[i] = sum(price_swaption_simul[0:i+1])/(i+1)*10000
print(f"price_swaption: {price_swaption_plot[-1]}")

# PLot of zcb prices, spot rates and instantaneous forward rates in the Vasicek model
fig = plt.figure(constrained_layout = False, dpi = 300, figsize = (5,3))
fig.suptitle(f"Spot, forward and par swap rates in the Vasicek model", fontsize = 10)
gs = fig.add_gridspec(nrows = 1, ncols = 1, left = 0.12, bottom = 0.2, right = 0.88, top = 0.90, wspace = 0, hspace = 0)
ax = fig.add_subplot(gs[0,0])
xticks = [0,2,4,6,8,10]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 6)
ax.set_xlim([xticks[0]-0.01,xticks[-1]+0.01])
plt.xlabel(f"Maturity",fontsize = 7)
yticks1 = [0,0.02,0.04,0.06,0.08]
ax.set_yticks(yticks1)
ax.set_yticklabels(yticks1,fontsize = 6)
ax.set_ylim([yticks1[0],yticks1[-1]+(yticks1[-1]-yticks1[0])*0.02])
plt.grid(axis = 'y', which='major', color=(0.7,0.7,0.7,0), linestyle='--')
ax.set_ylabel(f"Rates",fontsize = 7)
p1 = ax.scatter(T, R, s = 2, color = 'red', marker = ".",label="Spot rate")
p2 = ax.scatter(T, f, s = 2, color = 'blue', marker = ".",label="Instantaneous forward rate")
p3 = ax.scatter(T_swap, R_swap, s = 2, color = 'green', marker = ".",label="Par swap rate")
ax2 = ax.twinx()
yticks2 = [0,0.25,0.5,0.75,1]
ax2.set_yticks(yticks2)
ax2.set_yticklabels(yticks2,fontsize = 6)
ax2.set_ylim([yticks2[0],yticks2[-1] + (yticks2[-1]-yticks2[0])*0.02])
ax2.set_ylabel(f"ZCB prices",fontsize = 7)
p4 = ax2.scatter(T, p, s = 2, color = 'black', marker = ".",label="ZCB prices")
plots = [p1,p2,p3,p4]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="upper right",fontsize = 5)

# PLot of the 10Y par swap rate
fig = plt.figure(constrained_layout = False, dpi = 300, figsize=(5,3))
fig.suptitle(f"10Y par swap rate and 3M forward rates",fontsize=10)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])
xticks = [0,2,4,6,8,10]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 6)
ax.set_xlim([xticks[0]-0.01,xticks[-1]+0.01])
plt.xlabel(f"Maturity",fontsize = 7)
yticks1 = [0,0.02,0.04,0.06,0.08]
ax.set_yticks(yticks1)
ax.set_yticklabels(yticks1,fontsize = 6)
ax.set_ylim([yticks1[0],yticks1[-1]+(yticks1[-1]-yticks1[0])*0.02])
plt.grid(axis = 'y', which='major', color=(0.7,0.7,0.7,0), linestyle='--')
ax.set_ylabel(f"Rates",fontsize = 7)
p1 = ax.scatter(T, f_3m, s = 2, color = 'blue', marker = ".",label="3M forward rates")
p2 = ax.plot(T, R_swap[-1]*np.ones([len(T)]), linewidth = 1, color = 'red', marker = "",label="10Y par swap rate")
plots = [p1,p2[0]]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="upper right",fontsize = 5)

# PLot of simulated short rates in the Vasicek model
fig = plt.figure(constrained_layout=False,dpi=300,figsize=(5,3))
fig.suptitle(f"Simulated spot rates in the Vasicek model",fontsize=10)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])
xticks = [0,2,4,6,8,10]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 6)
ax.set_xlim([xticks[0]-0.01,xticks[-1]+0.01])
plt.xlabel(f"Time",fontsize = 7)
yticks1 = [0,0.02,0.04,0.06,0.08,0.1]
ax.set_yticks(yticks1)
ax.set_yticklabels(yticks1,fontsize = 6)
ax.set_ylim([yticks1[0],yticks1[-1] + (yticks1[-1]-yticks1[0])*0.02])
plt.grid(axis = 'y', which='major', color=(0.7,0.7,0.7,0), linestyle='--')
ax.set_ylabel(f"Short rate",fontsize = 7)
p1 = ax.scatter(t_simul, r_simul, s = 1, color = 'black', marker = ".",label="Simulated short rate")
p2 = ax.scatter(t_simul, lb, s = 1, color = 'red', marker = ".",label=f"Lower {size_ci} CB")
p3 = ax.scatter(t_simul, ub, s = 1, color = 'red', marker = ".",label=f"Upper {size_ci} CB")
p4 = ax.scatter(t_simul, lb_sd*np.ones([M_simul+1]), s = 1, color = 'blue', marker = ".",label=f"Lower {size_ci} CB Stat. dist.")
p5 = ax.scatter(t_simul, ub_sd*np.ones([M_simul+1]), s = 1, color = 'blue', marker = ".",label=f"Upper {size_ci} CB Stat. dist.")
plots = [p1,p2,p3,p4,p5]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="upper right",fontsize = 5)

# Plot of option prices as a function of N_simul
fig = plt.figure(constrained_layout=False,dpi=300,figsize=(5,3))
fig.suptitle(f"Swaption price as a function of number of simulations",fontsize=10)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])
xticks = [0,int((1/5)*N_simul),int((2/5)*N_simul),int((3/5)*N_simul),int((4/5)*N_simul),int(N_simul)]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 6)
ax.set_xlim([xticks[0]-2,xticks[-1]+2])
plt.xlabel(f"Number of simulations",fontsize = 7)
yticks1 = [0,50,100,150,200]
ax.set_yticks(yticks1)
ax.set_yticklabels(yticks1,fontsize = 6)
ax.set_ylim([yticks1[0],yticks1[-1] + (yticks1[-1]-yticks1[0])*0.02])
plt.grid(axis = 'y', which='major', color=(0.7,0.7,0.7,0), linestyle='--')
ax.set_ylabel(f"Swaption price",fontsize = 7)
p1 = ax.scatter([i for i in range(1,N_simul+1)], price_swaption_plot, s = 1, color = 'black', marker = ".",label=f"Swaption price: {np.round(price_swaption_plot[-1],2)}")
plots = [p1]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="upper right",fontsize = 5)
plt.show()
