import numpy as np
import fixed_income_derivatives_E2024 as fid
import matplotlib.pyplot as plt

EURIBOR_fixing = [{"id": 0,"instrument": "libor","maturity": 1/2, "rate":0.00967}]
fra_market = [{"id": 1,"instrument": "fra","exercise": 1/12,"maturity": 7/12, "rate": 0.00980},
{"id": 2,"instrument": "fra","exercise": 2/12,"maturity": 8/12, "rate": 0.01043},
{"id": 3,"instrument": "fra","exercise": 3/12,"maturity": 9/12, "rate": 0.01130},
{"id": 4,"instrument": "fra","exercise": 4/12,"maturity": 10/12, "rate": 0.01217},
{"id": 5,"instrument": "fra","exercise": 5/12,"maturity": 11/12, "rate": 0.01317},
{"id": 6,"instrument": "fra","exercise": 6/12,"maturity": 12/12, "rate": 0.01399},
{"id": 7,"instrument": "fra","exercise": 7/12,"maturity": 13/12, "rate": 0.01478},
{"id": 8,"instrument": "fra","exercise": 8/12,"maturity": 14/12, "rate": 0.01560},
{"id": 9,"instrument": "fra","exercise": 9/12,"maturity": 15/12, "rate": 0.01637}]
swap_market = [{"id": 10,"instrument": "swap","maturity": 2, "rate": 0.01652, "float_freq": "semiannual", "fixed_freq": "annual","indices": []},
{"id": 11,"instrument": "swap","maturity": 3, "rate": 0.02019, "float_freq": "semiannual", "fixed_freq": "annual","indices": []},
{"id": 12,"instrument": "swap","maturity": 4, "rate": 0.02319, "float_freq": "semiannual", "fixed_freq": "annual","indices": []},
{"id": 13,"instrument": "swap","maturity": 5, "rate": 0.02577, "float_freq": "semiannual", "fixed_freq": "annual","indices": []},
{"id": 14,"instrument": "swap","maturity": 7, "rate": 0.02995, "float_freq": "semiannual", "fixed_freq": "annual","indices": []},
{"id": 15,"instrument": "swap","maturity": 10, "rate": 0.03395, "float_freq": "semiannual", "fixed_freq": "annual","indices": []},
{"id": 16,"instrument": "swap","maturity": 15, "rate": 0.03753, "float_freq": "semiannual", "fixed_freq": "annual","indices": []},
{"id": 17,"instrument": "swap","maturity": 20, "rate": 0.03873, "float_freq": "semiannual", "fixed_freq": "annual","indices": []},
{"id": 18,"instrument": "swap","maturity": 30, "rate": 0.03975, "float_freq": "semiannual", "fixed_freq": "annual","indices": []}]
data = EURIBOR_fixing + fra_market + swap_market

interpolation_options = {"method":"hermite","degree":2}
T_fit, R_fit = fid.zcb_curve_fit(data,interpolation_options = interpolation_options)
p_inter, R_inter, f_inter, T_inter = fid.zcb_curve_interpolate(T_fit,R_fit,interpolation_options = interpolation_options,resolution = 1)

# 5Y interest rate swap before shift
T_5Y_swap_fixed_leg = np.array([i for i in range(0,6)])
p_5Y_swap_fixed_leg = np.array(fid.for_values_in_list_find_value_return_value(T_5Y_swap_fixed_leg,T_inter,p_inter))
S_5Y_swap = 0
for i in range(1,len(T_5Y_swap_fixed_leg)):
    S_5Y_swap += (T_5Y_swap_fixed_leg[i] - T_5Y_swap_fixed_leg[i-1])*p_5Y_swap_fixed_leg[i]
R_5Y_swap = (1-p_5Y_swap_fixed_leg[-1])/S_5Y_swap
print(f"5Y swap before changes. S_5Y_swap: {S_5Y_swap}, R_5Y_swap: {R_5Y_swap}")

# Shifting the entire spot rate zcb_curve
R_inter_zcb_shift = R_inter + np.array([0] + [0.0001 for i in range(1,len(R_inter))])
p_inter_zcb_shift = fid.zcb_prices_from_spot_rates(T_inter,R_inter_zcb_shift)
p_5Y_swap_fixed_leg_zcb_shift = np.array(fid.for_values_in_list_find_value_return_value(T_5Y_swap_fixed_leg,T_inter,p_inter_zcb_shift))
S_5Y_swap_zcb_shift = 0
for i in range(1,len(T_5Y_swap_fixed_leg)):
    S_5Y_swap_zcb_shift += (T_5Y_swap_fixed_leg[i] - T_5Y_swap_fixed_leg[i-1])*p_5Y_swap_fixed_leg_zcb_shift[i]
R_5Y_swap_zcb_shift = (1-p_5Y_swap_fixed_leg_zcb_shift[-1])/S_5Y_swap_zcb_shift
print(f"5Y swap after shift in all ZCB spot rates. S_5Y_swap: {S_5Y_swap_zcb_shift}, R_5Y_swap: {R_5Y_swap_zcb_shift}, DV01: {S_5Y_swap_zcb_shift*(R_5Y_swap-R_5Y_swap_zcb_shift)*100}")

# Shifting only the 5Y zcb rate
R_bump_5Y = np.zeros([len(R_inter)])
R_bump_5Y[60] = 0.0001
R_inter_5Y_shift = R_inter + R_bump_5Y
p_inter_5Y_shift = fid.zcb_prices_from_spot_rates(T_inter,R_inter_5Y_shift)
p_5Y_swap_fixed_leg_5Y_shift = np.array(fid.for_values_in_list_find_value_return_value(T_5Y_swap_fixed_leg,T_inter,p_inter_5Y_shift))
S_5Y_swap_5Y_shift = 0
for i in range(1,len(T_5Y_swap_fixed_leg)):
    S_5Y_swap_5Y_shift += (T_5Y_swap_fixed_leg[i] - T_5Y_swap_fixed_leg[i-1])*p_5Y_swap_fixed_leg_5Y_shift[i]
R_5Y_swap_5Y_shift = (1-p_5Y_swap_fixed_leg_5Y_shift[-1])/S_5Y_swap_5Y_shift
print(f"5Y swap after shift in only 5Y ZCB spot rate. S_5Y_swap: {S_5Y_swap_5Y_shift}, R_5Y_swap: {R_5Y_swap_5Y_shift}, DV01: {S_5Y_swap_5Y_shift*(R_5Y_swap-R_5Y_swap_5Y_shift)*100}")




# Shifting the 5Y swap market rate
EURIBOR_fixing = [{"id": 0,"instrument": "libor","maturity": 1/2, "rate":0.00967}]
fra_market = [{"id": 1,"instrument": "fra","exercise": 1/12,"maturity": 7/12, "rate": 0.00980},
{"id": 2,"instrument": "fra","exercise": 2/12,"maturity": 8/12, "rate": 0.01043},
{"id": 3,"instrument": "fra","exercise": 3/12,"maturity": 9/12, "rate": 0.01130},
{"id": 4,"instrument": "fra","exercise": 4/12,"maturity": 10/12, "rate": 0.01217},
{"id": 5,"instrument": "fra","exercise": 5/12,"maturity": 11/12, "rate": 0.01317},
{"id": 6,"instrument": "fra","exercise": 6/12,"maturity": 12/12, "rate": 0.01399},
{"id": 7,"instrument": "fra","exercise": 7/12,"maturity": 13/12, "rate": 0.01478},
{"id": 8,"instrument": "fra","exercise": 8/12,"maturity": 14/12, "rate": 0.01560},
{"id": 9,"instrument": "fra","exercise": 9/12,"maturity": 15/12, "rate": 0.01637}]
swap_market_mr_shift = [{"id": 10,"instrument": "swap","maturity": 2, "rate": 0.01652, "float_freq": "semiannual", "fixed_freq": "annual","indices": []},
{"id": 11,"instrument": "swap","maturity": 3, "rate": 0.02019, "float_freq": "semiannual", "fixed_freq": "annual","indices": []},
{"id": 12,"instrument": "swap","maturity": 4, "rate": 0.02319, "float_freq": "semiannual", "fixed_freq": "annual","indices": []},
{"id": 13,"instrument": "swap","maturity": 5, "rate": 0.02577 + 0.0001, "float_freq": "semiannual", "fixed_freq": "annual","indices": []},
{"id": 14,"instrument": "swap","maturity": 7, "rate": 0.02995, "float_freq": "semiannual", "fixed_freq": "annual","indices": []},
{"id": 15,"instrument": "swap","maturity": 10, "rate": 0.03395, "float_freq": "semiannual", "fixed_freq": "annual","indices": []},
{"id": 16,"instrument": "swap","maturity": 15, "rate": 0.03753, "float_freq": "semiannual", "fixed_freq": "annual","indices": []},
{"id": 17,"instrument": "swap","maturity": 20, "rate": 0.03873, "float_freq": "semiannual", "fixed_freq": "annual","indices": []},
{"id": 18,"instrument": "swap","maturity": 30, "rate": 0.03975, "float_freq": "semiannual", "fixed_freq": "annual","indices": []}]
data_mr_shift = EURIBOR_fixing + fra_market + swap_market_mr_shift

T_fit_mr_shift, R_fit_mr_shift = fid.zcb_curve_fit(data_mr_shift,interpolation_options = interpolation_options)
p_inter_mr_shift, R_inter_mr_shift, f_inter_mr_shift, T_inter_mr_shift = fid.zcb_curve_interpolate(T_fit_mr_shift,R_fit_mr_shift,interpolation_options = interpolation_options,resolution = 1)
p_5Y_swap_fixed_leg_mr_shift = np.array(fid.for_values_in_list_find_value_return_value(T_5Y_swap_fixed_leg,T_inter_mr_shift,p_inter_mr_shift))
S_5Y_swap_mr_shift = 0
for i in range(1,len(T_5Y_swap_fixed_leg)):
    S_5Y_swap_mr_shift += (T_5Y_swap_fixed_leg[i] - T_5Y_swap_fixed_leg[i-1])*p_5Y_swap_fixed_leg_mr_shift[i]
R_5Y_swap_mr_shift = (1-p_5Y_swap_fixed_leg_mr_shift[-1])/S_5Y_swap_mr_shift
print(f"5Y swap after shift in 5Y swap market rate. S_5Y_swap: {S_5Y_swap_mr_shift}, R_5Y_swap: {R_5Y_swap_mr_shift}, DV01: {S_5Y_swap_mr_shift*(R_5Y_swap-R_5Y_swap_mr_shift)*100}")

fig = plt.figure(constrained_layout=False, dpi = 300, figsize = (5,3))
fig.suptitle(f"Calibrated zero coupon spot rates", fontsize = 9)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])
xticks = [0,1,2,3,4,5,7,10,15,20,30]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 6)
ax.set_xlim([xticks[0]+-0.2,xticks[-1]+0.2])
plt.xlabel(f"Maturity",fontsize = 7)
ax.set_yticks([0,0.01,0.02,0.03,0.04,0.05,0.06])
ax.set_yticklabels([0,0.01,0.02,0.03,0.04,0.05,0.06],fontsize = 6)
ax.set_ylim([0,0.0625])
ax.set_ylabel(f"Spot and instantaneous forward rates",fontsize = 7)
plt.grid(axis = 'y', which='major', color = (0.7,0.7,0.7,0), linestyle='--')
p1 = ax.scatter(T_inter[1:], R_inter[1:], s = 1, color = 'black', marker = ".",label="Spot rates")
p2 = ax.scatter(T_inter[1:], f_inter[1:], s = 1, color = 'red', marker = ".",label="forward rates")
plots = [p1,p2]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="lower right",fontsize = 6)
fig.savefig("C:/Jacob/Uni_of_CPH/FID/FID_E2024/Lecture Notes/Lecture_slides_II_static_term_structure_models/yield_curve_fit.pdf")
# plt.show()
