import numpy as np
from scipy.optimize import minimize
import fixed_income_derivatives_E2024 as fid
import matplotlib.pyplot as plt
#from numpy.polynomial.hermite import hermfit, hermval
import copy

EURIBOR_fixing = [{"id": 0,"instrument": "libor","maturity": 1/2, "rate":0.03478}]
fra_market = [{"id": 1,"instrument": "fra","exercise": 1/12,"maturity": 7/12, "rate": 0.03743},
{"id": 2,"instrument": "fra","exercise": 2/12,"maturity": 8/12, "rate": 0.03871},
{"id": 3,"instrument": "fra","exercise": 3/12,"maturity": 9/12, "rate": 0.03989},
{"id": 4,"instrument": "fra","exercise": 4/12,"maturity": 10/12, "rate": 0.04098},
{"id": 5,"instrument": "fra","exercise": 5/12,"maturity": 11/12, "rate": 0.04198},
{"id": 6,"instrument": "fra","exercise": 6/12,"maturity": 12/12, "rate": 0.04289},
{"id": 7,"instrument": "fra","exercise": 7/12,"maturity": 13/12, "rate": 0.04374},
{"id": 8,"instrument": "fra","exercise": 8/12,"maturity": 14/12, "rate": 0.04452},
{"id": 9,"instrument": "fra","exercise": 9/12,"maturity": 15/12, "rate": 0.04524}]
swap_market = [{"id": 10,"instrument": "swap","maturity": 2, "rate": 0.04377, "float_freq": "semiannual", "fixed_freq": "annual","indices": []},
{"id": 11,"instrument": "swap","maturity": 3, "rate": 0.04625, "float_freq": "semiannual", "fixed_freq": "annual","indices": []},
{"id": 12,"instrument": "swap","maturity": 4, "rate": 0.04777, "float_freq": "semiannual", "fixed_freq": "annual","indices": []},
{"id": 13,"instrument": "swap","maturity": 5, "rate": 0.04875, "float_freq": "semiannual", "fixed_freq": "annual","indices": []},
{"id": 14,"instrument": "swap","maturity": 7, "rate": 0.04992, "float_freq": "semiannual", "fixed_freq": "annual","indices": []},
{"id": 15,"instrument": "swap","maturity": 10, "rate": 0.05081, "float_freq": "semiannual", "fixed_freq": "annual","indices": []},
{"id": 16,"instrument": "swap","maturity": 15, "rate": 0.05148, "float_freq": "semiannual", "fixed_freq": "annual","indices": []},
{"id": 17,"instrument": "swap","maturity": 20, "rate": 0.05181, "float_freq": "semiannual", "fixed_freq": "annual","indices": []},
{"id": 18,"instrument": "swap","maturity": 30, "rate": 0.05211, "float_freq": "semiannual", "fixed_freq": "annual","indices": []}]
data = EURIBOR_fixing + fra_market + swap_market

# Problem 1 - Fitting the yield curve
interpolation_options = {"method":"nelson_siegel","transition": "smooth"}
interpolation_options = {"method":"hermite","degree":2,"transition": "smooth"}
T_fit, R_fit = fid.zcb_curve_fit(data,interpolation_options = interpolation_options)
T_6m = np.array([i*0.5 for i in range(0,61)])
p_inter, R_inter, f_inter, T_inter = fid.zcb_curve_interpolate(T_6m,T_fit,R_fit,interpolation_options = interpolation_options)

# c) The par swap rate curve
T_swap = np.array([i for i in range(1,31)])
R_swap, S_swap = np.zeros([len(T_swap)]), np.zeros([len(T_swap)])
for i, maturity in enumerate(T_swap):
    R_swap[i], S_swap[i] = fid.swap_rate_from_zcb_prices(0,0,T_swap[i],"annual",T_inter,p_inter)
print(f"10Y par swap rate: {R_swap[9]}, accrual factor: {S_swap[9]}")

# Problem 2 - Bumping spot rates and finding the DV01 of a swap
swap_id, size_bump = 15, 0.0001
print(f"Bumping spot rates")
def dv01_swap_spot_rate_bump_fct(t,T_n,T_N,fixed_freq,R_swap_init,T_bump,size_bump,T_inter,R_inter,p_inter):
    R_bump, p_bump = fid.spot_rate_bump(T_bump,size_bump,T_inter,R_inter,p_inter)
    R_swap_bump, S_swap_bump = fid.swap_rate_from_zcb_prices(t,T_n,T_N,fixed_freq,T_inter,p_bump)
    DV01 = (R_swap_bump-R_swap_init)*S_swap_bump
    return DV01

# 2a) DV01 when bumping a single spot rate
idx_bump_single = 20
R_bump = R_inter.copy()
R_bump[idx_bump_single] += size_bump
p_bump = fid.zcb_prices_from_spot_rates(T_inter,R_bump)
R_swap_bump, S_swap_bump = fid.swap_rate_from_zcb_prices(0,0,data[swap_id]["maturity"],"annual",T_inter,p_bump)
print(f"R_swap_bump: {R_swap_bump}, S_swap_bump: {S_swap_bump}")
DV01 = (R_swap_bump-data[swap_id]["rate"])*S_swap_bump
print(f"DV01 for swap {swap_id} when bumping spot_rate for T: {T_inter[idx_bump_single]} is: {10000*DV01}")
# 2a) DV01 for bumping each of the spot rates at T=[1,2,3,4,5,6,7,8,9,10]
DV01_bump = np.zeros([10])
T_bump = np.array([i for i in range(1,11)])
for i, val in enumerate(T_bump):
    DV01_bump[i] = dv01_swap_spot_rate_bump_fct(0,0,data[swap_id]["maturity"],"annual",data[swap_id]["rate"],val,size_bump,T_inter,R_inter,p_inter)
print(f"DV01 when bumping each spot rate separately: {10000*DV01_bump}")
# 2b) DV01 for bumping all of the spot rates at T=[1,2,3,4,5,6,7,8,9,10]
R_bump, p_bump = fid.spot_rate_bump(T_bump,size_bump,T_inter,R_inter,p_inter)
R_swap_bump, S_swap_bump = fid.swap_rate_from_zcb_prices(0,0,data[swap_id]["maturity"],"annual",T_inter,p_bump)
print(f"R_swap_bump: {R_swap_bump}, S_swap_bump: {S_swap_bump}")
DV01 = (R_swap_bump-data[swap_id]["rate"])*S_swap_bump
print(f"DV01 for swap {swap_id} when bumping spot_rates for T: {T_bump} is: {10000*DV01}")

# Problem 3 - Bumping market rates and finding the DV01 of a swap
print(f"Bumping market rates")
def dv01_market_rate_bump_fct(t,T_n,T_N,T_inter,fixed_freq,R_swap_init,idx_bump,size_bump,data,interpolation_options):
    p_bump, R_bump, f_bump, T_bump, data_bump = fid.market_rate_bump(idx_bump,size_bump,T_inter,data,interpolation_options = interpolation_options)
    R_swap_bump, S_swap_bump = fid.swap_rate_from_zcb_prices(0,0,T_N,fixed_freq,T_bump,p_bump)
    DV01 = (R_swap_bump-R_swap_init)*S_swap_bump
    return DV01

# DV01 when bumping a single market rate
idx_bump_single = 15
p_bump, R_bump, f_bump, T_bump, data_bump = fid.market_rate_bump(idx_bump_single,size_bump,T_inter,data,interpolation_options = interpolation_options)
R_swap_bump, S_swap_bump = fid.swap_rate_from_zcb_prices(0,0,data[swap_id]["maturity"],"annual",T_inter,p_bump)
print(f"R_swap_bump: {R_swap_bump}, S_swap_bump: {S_swap_bump}")
DV01 = (R_swap_bump-data[swap_id]["rate"])*S_swap_bump
print(f"DV01 for swap {swap_id} when bumping market rates for idx: {idx_bump_single} is: {10000*DV01}")
# DV01 when bumping each of the market rates
idx_bump_all = np.array([i for i in range(0,19)])
DV01_bump = np.zeros([19])
for i, idx in enumerate(idx_bump_all):
    DV01_bump[i] = dv01_market_rate_bump_fct(0,0,data[swap_id]["maturity"],T_inter,"annual",data[swap_id]["rate"],idx,size_bump,data,interpolation_options)
print(f"DV01 when bumping each market rate separately: {10000*DV01_bump}")
# DV01 when bumping all of the market rates
p_bump, R_bump, f_bump, T_bump, data_bump = fid.market_rate_bump(idx_bump_all,size_bump,T_inter,data,interpolation_options = interpolation_options)
R_swap_bump, S_swap_bump = fid.swap_rate_from_zcb_prices(0,0,data[swap_id]["maturity"],"annual",T_inter,p_bump)
print(f"R_swap_bump: {R_swap_bump}, S_swap_bump: {S_swap_bump}")
DV01 = (R_swap_bump-data[swap_id]["rate"])*S_swap_bump
print(f"DV01 for swap {swap_id} when bumping spot_rates for idx: {idx_bump_all} is: {10000*DV01}")

# Problem 4 - Mark-to-market of Swap Positions
T_pos, R_pos = 6, 0.048
swaption_price = 38  # price in bps
R_swap_pos, S_swap_pos = fid.swap_rate_from_zcb_prices(0,0,T_pos,"annual",T_inter,p_inter)
PnL = (R_swap_pos-R_pos)*S_swap_pos
print(f"6Y par swap rate: {R_swap_pos}, S_swap: {S_swap_pos}, PnL of position in 7Y payer swap after one year: {10000*PnL} in bps.")
print(f"Market value of payer swaption now at exercise: {10000*PnL}, PnL of position in swaption: {10000*PnL-swaption_price}.")
T_n, T_N, R_pos2 = 3, 8, 0.051
R_swap_pos2, S_swap_pos2 = fid.swap_rate_from_zcb_prices(0,T_n,T_N,"annual",T_inter,p_inter)
PnL2 = (R_pos2 - R_swap_pos2)*S_swap_pos2
print(f"3Y5Y forward par swap rate: {R_swap_pos2}, S_swap_pos: {S_swap_pos2}, PnL of position in : {10000*PnL2}.")

# Problem 5 - Fitting a Vasicek model to the yield curve
param_0 = 0.03, 0.5, 0.04, 0.03
result = minimize(fid.fit_vasicek_obj,param_0,method = 'nelder-mead',args = (R_inter,T_inter),options={'xatol': 1e-20,'disp': True})
print(f"Parameters from the fit: {result.x}. SSE of the fit: {result.fun}")
r0, a, b, sigma = result.x
p_vasicek = fid.zcb_price_vasicek(r0,a,b,sigma,T_inter)
f_vasicek = fid.forward_rate_vasicek(r0,a,b,sigma,T_inter)
R_vasicek = fid.spot_rate_vasicek(r0,a,b,sigma,T_inter)

fig = plt.figure(constrained_layout=False, dpi = 300, figsize = (5,3))
fig.suptitle(f"Calibrated zero coupon spot rates", fontsize = 9)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])
xticks = [0,1,2,3,4,5,7,10,15,20,30]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 6)
ax.set_xlim([xticks[0]+-0.2,xticks[-1]+0.2])
plt.xlabel(f"Maturity",fontsize = 6)
ax.set_yticks([0,0.01,0.02,0.03,0.04,0.05,0.06])
ax.set_yticklabels([0,0.01,0.02,0.03,0.04,0.05,0.06],fontsize = 6)
ax.set_ylim([0,0.0625])
plt.grid(axis = 'y', which='major', color=(0.7,0.7,0.7,0), linestyle='--')
p1 = ax.scatter(T_inter, R_inter, s = 1, color = 'black', marker = ".",label="Spot rates")
p2 = ax.scatter(T_inter, f_inter, s = 1, color = 'red', marker = ".",label="forward rates")
p3 = ax.scatter(T_swap, R_swap, s = 1, color = 'green', marker = ".",label="par swap rates")
plots = [p1,p2,p3]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="lower right",fontsize = 6)
fig.savefig("C:/Jacob/Uni_of_CPH/FID/FID_E2024/Examples/curve_fit_zcb_fit.pdf")
plt.show()

fig = plt.figure(constrained_layout=False, dpi = 300, figsize = (5,3))
fig.suptitle(f"Calibrated zero coupon spot rates", fontsize = 9)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])
xticks = [0,1,2,3,4,5,7,10,15,20,30]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 6)
ax.set_xlim([xticks[0]+-0.2,xticks[-1]+0.2])
plt.xlabel(f"Maturity",fontsize = 6)
ax.set_yticks([0,0.01,0.02,0.03,0.04,0.05,0.06])
ax.set_yticklabels([0,0.01,0.02,0.03,0.04,0.05,0.06],fontsize = 6)
ax.set_ylim([0,0.0625])
plt.grid(axis = 'y', which='major', color=(0.7,0.7,0.7,0), linestyle='--')
p1 = ax.scatter(T_inter, R_inter, s = 1, color = 'black', marker = ".",label="Spot rates")
p2 = ax.scatter(T_inter, f_inter, s = 1, color = 'red', marker = ".",label="forward rates")
p3 = ax.scatter(T_swap, R_swap, s = 1, color = 'green', marker = ".",label="par swap rates")
p4 = ax.scatter(T_inter, R_vasicek, s = 1, color = 'blue', marker = ".",label="Vasicek Spot rates")
p5 = ax.scatter(T_inter, f_vasicek, s = 1, color = 'orange', marker = ".",label="Vasicek forward rates")
plots = [p1,p2,p3,p4,p5]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="lower right",fontsize = 6)
fig.savefig("C:/Jacob/Uni_of_CPH/FID/FID_E2024/Examples/curve_fit_zcb_fit_vasicek_fit.pdf")
plt.show()
