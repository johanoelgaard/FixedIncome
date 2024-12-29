import numpy as np
from scipy.stats import norm, ncx2, gamma
import matplotlib.pyplot as plt
import fixed_income_derivatives as fid

# Problem 1
# r0, a, b, sigma = 0.025, 0.5, 0.025, 0.02
# N, T = 200, 10
# mesh = T/N
#
# tau = [i*mesh for i in range(0,N+1)]
# p = fid.zcb_price_vasicek(r0, a, b, sigma, tau)
# R = fid.spot_rate_vasicek(r0, a, b, sigma, tau)
# f = fid.forward_rate_vasicek(r0, a, b, sigma, tau)
# # lb, ub = ci_vasicek(r0,a,b,sigma,tau,size_ci,method = "two_sided")
#
# fig = plt.figure(constrained_layout=False, dpi = 300, figsize = (5,3))   #
# fig.suptitle(f"ZCB prices, spot rates and forward rates for Vasicek", fontsize = 9)
# gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
# ax = fig.add_subplot(gs[0,0])
#
# xticks = [0,2,4,6,8,10]
# ax.set_xticks(xticks)
# ax.set_xticklabels(xticks,fontsize = 6)
# ax.set_xlim([xticks[0]-0.2,xticks[-1]+0.2])
# plt.xlabel(f"Maturity",fontsize = 6)
# ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
# ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1],fontsize = 6)
# ax.set_ylim([0,1.05])
# ax.set_ylabel(f"ZCB prices",fontsize = 6)
# p1 = ax.scatter(tau, p, s = 1, color = 'black', marker = ".",label="p")
#
# ax1 = ax.twinx()
# ax1.set_yticks([0,0.02,0.04,0.06,0.08,0.1])
# ax1.set_yticklabels([0,0.02,0.04,0.06,0.08,0.1],fontsize = 6)
# ax1.set_ylim([0,0.1025])
# p2 = ax1.plot(tau, (b/a)*np.ones([N+1]), color = 'black', marker = "", linewidth = 1, linestyle = "solid",label = "b/a")
# p3 = ax1.scatter(tau, R, s = 1, color = 'red', marker = ".",label="R")
# p4 = ax1.scatter(tau, f, s = 1, color = 'blue', marker = ".",label="f")
# # p4 = ax1.scatter(tau, b/a*np.ones([N+1]), s = 0.25, color = "green", marker = ".",label="b/a")
#
# plots = [p1,p2[0],p3,p4]
# labels = [item.get_label() for item in plots]
# ax.legend(plots,labels,loc="upper right",fontsize = 6)
#
# plt.show()

# Problem 2
# r0, a, b, sigma = 0.025, 2, 0.05, 0.1
# N, T = 200, 10
# size_ci = 0.95
# mesh = T/N
#
# tau = [i*mesh for i in range(0,N+1)]
# p = fid.zcb_price_cir(r0, a, b, sigma, tau)
# R = fid.spot_rate_cir(r0, a, b, sigma, tau)
# f = fid.forward_rate_cir(r0,a,b,sigma,tau)
#
# fig = plt.figure(constrained_layout=False, dpi = 300, figsize = (5,3))   #
# fig.suptitle(f"ZCB prices, spot rates and forward rates for CIR", fontsize = 9)
# gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
# ax = fig.add_subplot(gs[0,0])
#
# xticks = [0,2,4,6,8,10]
# ax.set_xticks(xticks)
# ax.set_xticklabels(xticks,fontsize = 6)
# ax.set_xlim([xticks[0]-0.2,xticks[-1]+0.2])
# plt.xlabel(f"Maturity",fontsize = 6)
# ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
# ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1],fontsize = 6)
# ax.set_ylim([0,1.05])
# ax.set_ylabel(f"ZCB prices",fontsize = 6)
# p1 = ax.scatter(tau, p, s = 1, color = 'black', marker = ".",label="p")
#
# ax1 = ax.twinx()
# ax1.set_yticks([0,0.02,0.04,0.06,0.08,0.1])
# ax1.set_yticklabels([0,0.02,0.04,0.06,0.08,0.1],fontsize = 6)
# ax1.set_ylim([0,0.1025])
# p2 = ax1.plot(tau, b*np.ones([N+1]), color = 'black', marker = "", linewidth = 1, linestyle = "solid",label = "b")
# p3 = ax1.scatter(tau, R, s = 1, color = 'red', marker = ".",label="R")
# p4 = ax1.scatter(tau, f, s = 1, color = 'blue', marker = ".",label="f")
# # p4 = ax1.scatter(tau, b*np.ones([N+1]), s = 0.25, color = "green", marker = ".",label="b")
#
# plots = [p1,p2[0],p3,p4]
# labels = [item.get_label() for item in plots]
# ax.legend(plots,labels,loc="upper right",fontsize = 6)
#
# plt.show()

# Problem 3
# r0, a, b, sigma = 0.025, 2, 0.1, 0.02
# size_ci = 0.95
# T, N = 10, 50000
#
# param = (a,b,sigma)
# delta = T/N
# r = fid.short_rate_simul(r0,param,N,T,method = "vasicek")
# t = np.array([i*delta for i in range(0,N+1)])
# lb,ub = fid.ci_vasicek(r0,a,b,sigma,t,size_ci,method = "two_sided")
#
# fig = plt.figure(constrained_layout=False, dpi = 300, figsize = (5,3))   #
# fig.suptitle(f"Simulation of the short rate in Vasicek", fontsize = 9)
# gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
# ax = fig.add_subplot(gs[0,0])
#
# xticks = [0,(1/5)*T,(2/5)*T,(3/5)*T,(4/5)*T,T]
# xticklabels = [round(i,3) for i in xticks]
# yticks = [0,0.02,0.04,0.06,0.08,0.1]
#
# ax.set_xticks(xticks)
# ax.set_xticklabels(xticklabels,fontsize = 6)
# ax.set_xlim([xticks[0]-0.2,xticks[-1]+0.2])
# plt.xlabel(f"time",fontsize = 6)
# ax.set_yticks(yticks)
# ax.set_yticklabels(yticks,fontsize = 6)
# ax.set_ylim([yticks[0],yticks[-1]*1.05])
# ax.set_ylabel(f"Short rate",fontsize = 6)
# ax.scatter(t, r, s = 1, color = 'black', marker = ".", label = "Short rate")
# lb_sd = (b/a-1.96*sigma/np.sqrt(2*a))*np.ones([N+1])
# ax.plot(t,lb_sd,color = 'blue',marker = "", linewidth = 1)
# ax.plot(t, lb, color = 'red', marker = "", linewidth = 1, linestyle = "dashed", label = "Lower CI")
# ub_sd = (b/a+1.96*sigma/np.sqrt(2*a))*np.ones([N+1])
# ax.plot(t,ub_sd,color = 'blue',marker = "", linewidth = 1)
# ax.plot(t, ub, color = 'red', marker = "", linewidth = 1, linestyle = "dashed", label = "Upper CI")
#
# ax.legend(loc = "upper right",fontsize = 6)
# plt.show()

# Problem 4
r0, a, b, sigma = 0.025, 2, 0.05, 0.1
T, N = 10, 50000
size_ci = 0.95

param = (a,b,sigma)
delta = T/N
t = np.array([i*delta for i in range(0,N+1)])
r = fid.short_rate_simul(r0,param,N,T,method = "cir")
N_ci = 100
t_ci = [i*T/N_ci for i in range(0,N_ci+1)]
lb, ub = fid.ci_cir(r0,a,b,sigma,t_ci,size_ci,method = "two_sided")

fig = plt.figure(constrained_layout=False, dpi = 300, figsize = (5,3))   #
fig.suptitle(f"Simulation of the short rate in CIR", fontsize = 9)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])

xticks = [0,(1/5)*T,(2/5)*T,(3/5)*T,(4/5)*T,T]
xticklabels = [round(i,3) for i in xticks]
yticks = [0,0.02,0.04,0.06,0.08,0.1]

ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels,fontsize = 6)
ax.set_xlim([xticks[0]-0.2,xticks[-1]+0.2])
plt.xlabel(f"time",fontsize = 6)
ax.set_yticks(yticks)
ax.set_yticklabels(yticks,fontsize = 6)
ax.set_ylim([yticks[0],yticks[-1]*1.05])
ax.set_ylabel(f"Short rate",fontsize = 6)
ax.scatter(t, r, s = 1, color = 'black', marker = ".",label = "short rate")
alpha, beta = (2*a*b)/(sigma**2), sigma**2/(2*a)
lb_sd, ub_sd = gamma.ppf(0.025, alpha, loc=0, scale=beta)*np.ones([N_ci+1]), gamma.ppf(0.975, alpha, loc=0, scale=beta)*np.ones([N_ci+1])
ax.plot(t_ci,lb_sd,color = 'blue',marker = "", linewidth = 1)
ax.plot(t_ci, lb, color = 'red', marker = "", linewidth = 1, linestyle = "dashed", label = "Lower CI")
# ax.scatter(t_ci, lb, s = 0.25, color = "green", marker = ".")
ax.plot(t_ci,ub_sd,color = 'blue',marker = "", linewidth = 1)
ax.plot(t_ci, ub, color = 'red', marker = "", linewidth = 1, linestyle = "dashed", label = "Upper CI")
# ax.scatter(t_ci, ub, s = 0.25, color = "green", marker = ".")

ax.legend(loc = "upper right",fontsize = 6)
plt.show()
