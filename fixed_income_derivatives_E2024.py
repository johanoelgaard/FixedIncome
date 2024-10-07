import numpy as np
from scipy.optimize import minimize
from numpy.polynomial.hermite import hermfit, hermval, hermder
import copy

def zcb_prices_from_spot_rates(T,spot_rate):
    M = len(T)
    p = np.zeros([M])
    for i in range(0,M):
        if T[i] < 1e-8:
            p[i] = 1
        else:
            p[i] = np.exp(-spot_rate[i]*T[i])
    return p

def spot_rates_from_zcb_prices(T,zcb_price):
    M = len(T)
    r = np.zeros([M])
    for i in range(0,M):
        if T[i] < 1e-12:
            r[i] = np.nan
        else:
            r[i] = -np.log(zcb_price[i])/T[i]
    return r



#  Fixed rate bond
def macauley_duration(pv,T,C,ytm):
    D = 0
    N = len(T)
    for i in range(0,N):
        D += C[i]*T[i]/(1+ytm)**T[i]
    D = D/pv
    return D

def modified_duration(pv,T,C,ytm):
    D = 0
    N = len(T)
    for i in range(0,N):
        D += C[i]*T[i]/(1+ytm)**T[i]
    D = D/(pv*(1+ytm))
    return D

def convexity(pv,T,C,ytm):
    D = 0
    N = len(T)
    for i in range(0,N):
        D += C[i]*T[i]**2/(1+ytm)**T[i]
    D = D/pv
    return D

def price_fixed_rate_bond_from_ytm(ytm,T,C):
    price = 0
    N = len(T)
    for i in range(0,N):
        price += C[i]/(1+ytm)**T[i]
    return price

def ytm(pv,T,C,ytm_init = 0.05):
    args = (pv, T, C, 1)
    result = minimize(ytm_obj,ytm_init,args = args, options={'disp': False})
    ytm = result.x[0]
    return ytm

def ytm_obj(ytm,pv,T,C,scaling = 1):
    N = len(T)
    pv_new = 0
    for i in range(0,N):
        pv_new += C[i]/(1+ytm[0])**T[i]
    sse = scaling*(pv-pv_new)**2
    return sse

# List operatins
def find_value_return_value(val,L1,L2,precision = 10e-8):
    # This function searches for 'val' in 'L1' and returns index 'idx' of 'val' in 'L1' and 'L2[idx]'.
    Ind, output = False, []
    for idx, item in enumerate(L1):
        if abs(val-item) < precision:
            Ind = True
            output.append((idx,L2[idx]))
    return Ind, output

def for_values_in_list_find_value_return_value(L1,L2,L3,precision = 10e-8):
    # For all 'item' in L1, this function searches for 'item' in L2 and returns the value corresponding to same index from 'L3'.
    output = len(L1)*[None]
    for i, item in enumerate(L1):
        Ind, output_temp = find_value_return_value(item,L2,L3,precision)
        if Ind == True:
            output[i] = output_temp[0][1]
    return output

# ZCB curvefitting
def zcb_curve_fit(data_input,interpolation_options = {"method": "linear"},scaling = 1):
    data = copy.deepcopy(data_input)
    data_known = []
    libor_data, fra_data, swap_data = [], [], []
    # Separateing the data and constructing data_known from fixings
    for item in data:
        if item["instrument"] == "libor":
            libor_data.append(item)
            data_known.append({"maturity":item["maturity"],"rate":np.log(1+item["rate"]*item["maturity"])/item["maturity"]})
        elif item["instrument"] == "fra":
            fra_data.append(item)
        elif item["instrument"] == "swap":
            swap_data.append(item)
    # Adding elements to data_knwon based on FRAs
    I_done = False
    while I_done == False:
        for fra in fra_data:
            I_exer, known_exer = value_in_list_of_dict_returns_I_idx(fra["exercise"],data_known,"maturity")
            I_mat, known_mat = value_in_list_of_dict_returns_I_idx(fra["maturity"],data_known,"maturity")
            # FIX the last few instances!!!!
            if I_exer == True and I_mat == False:
                data_known.append({"maturity":fra["maturity"],"rate":(known_exer["rate"]*known_exer["maturity"]+np.log(1+(fra["maturity"]-fra["exercise"])*fra["rate"]))/fra["maturity"]})
                I_done = False
                break
            if I_exer == False and I_mat == True:
                pass
            if I_exer == True and I_mat == True:
                pass
            else:
                I_done = True
    T_known, T_swap, T_knot = [], [], []
    R_known = []
    # Finding T's and corresponding R's where there is some known data
    for known in data_known:
        T_known.append(known["maturity"])
        R_known.append(known["rate"])
    for swap in swap_data:
        T_knot.append(swap["maturity"])
        if swap["float_freq"] == "quarterly":
            if value_in_list_returns_I_idx(0.25,T_known)[0] == False and value_in_list_returns_I_idx(0.25,T_swap)[0] == False:
                T_swap.append(0.25)
        elif swap["float_freq"] == "semiannual":
            if value_in_list_returns_I_idx(0.5,T_known)[0] == False and value_in_list_returns_I_idx(0.5,T_swap)[0] == False:
                T_swap.append(0.5)
        elif swap["float_freq"] == "annual":
            if value_in_list_returns_I_idx(1,T_known)[0] == False and value_in_list_returns_I_idx(1,T_swap)[0] == False:
                T_swap.append(1)
        if swap["fixed_freq"] == "quarterly":
            for i in range(1,4*swap["maturity"]):
                if value_in_list_returns_I_idx(i*0.25,T_known)[0] == False and value_in_list_returns_I_idx(i*0.25,T_knot)[0] == False and value_in_list_returns_I_idx(i*0.25,T_swap)[0] == False:
                    T_swap.append(i*0.25)
        elif swap["fixed_freq"] == "semiannual":
            for i in range(1,2*swap["maturity"]):
                if value_in_list_returns_I_idx(i*0.5,T_known)[0] == False and value_in_list_returns_I_idx(i*0.5,T_knot)[0] == False and value_in_list_returns_I_idx(i*0.5,T_swap)[0] == False:
                    T_swap.append(i*0.5)
        elif swap["fixed_freq"] == "annual":
            for i in range(1,swap["maturity"]):
                if value_in_list_returns_I_idx(i,T_known)[0] == False and value_in_list_returns_I_idx(i*1,T_knot)[0] == False and value_in_list_returns_I_idx(i,T_swap)[0] == False:
                    T_swap.append(i)
    # Finding T_fra and T_endo
    T_endo, T_fra = [], []
    fra_data.reverse()
    for fra in fra_data:
        if value_in_list_returns_I_idx(fra["maturity"],T_known)[0] == False:
            I_fra_mat, idx_fra_mat = value_in_list_returns_I_idx(fra["maturity"],T_fra)
            I_endo_mat, idx_endo_mat = value_in_list_returns_I_idx(fra["maturity"],T_endo)
            if I_fra_mat is False and I_endo_mat is False:
                T_fra.append(fra["maturity"])
            elif I_fra_mat is True and I_endo_mat is False:
                pass
            elif I_fra_mat is False and I_endo_mat is True:
                pass
            elif I_fra_mat is True and I_endo_mat is True:
                T_fra.pop(idx_fra_mat)
        if value_in_list_returns_I_idx(fra["exercise"],T_known)[0] == False:
            I_fra_exer, idx_fra_exer = value_in_list_returns_I_idx(fra["exercise"],T_fra)
            I_endo_exer, idx_endo_exer = value_in_list_returns_I_idx(fra["exercise"],T_endo)
            if I_fra_exer is False and I_endo_exer is False:
                T_endo.append(fra["exercise"])
            elif I_fra_exer is True and I_endo_exer is False:
                T_fra.pop(idx_fra_exer)
                T_endo.append(fra["exercise"])
            elif I_fra_exer is False and I_endo_exer is True:
                pass
            elif I_fra_exer is True and I_endo_exer is True:
                T_fra.pop(idx_fra_exer)
    fra_data.reverse()
    # Fitting the swap portino of the curve
    T_swap_fit = T_known + T_swap + T_knot
    T_swap_fit.sort(), T_fra.sort(), T_endo.sort()
    R_knot_init = [None]*len(swap_data)
    for i, swap in enumerate(swap_data):
        indices = []
        if swap["fixed_freq"] == "quarterly":
            for j in range(1,4*swap["maturity"]+1):
                indices.append(value_in_list_returns_I_idx(j*0.25,T_swap_fit)[1])
        elif swap["fixed_freq"] == "semiannual":
            for j in range(1,2*swap["maturity"]+1):
                indices.append(value_in_list_returns_I_idx(j*0.5,T_swap_fit)[1])
        elif swap["fixed_freq"] == "annual":
            for j in range(1,swap["maturity"]+1):
                indices.append(value_in_list_returns_I_idx(j,T_swap_fit)[1])
        swap["indices"] = indices
        R_knot_init[i] = swap["rate"]
        i += 1
    args = (T_known,T_knot,T_swap_fit,R_known,swap_data,interpolation_options,1)
    result = minimize(zcb_curve_swap_fit_obj,R_knot_init,args = args,options={'disp': False})
    T_swap_curve, R_swap_curve = T_known + T_knot, R_known + list(result.x)
    T_fra_fit = T_swap_curve + T_fra + T_endo
    T_fra_fit.sort()
    R_fra_fit = interpolate(T_swap_curve,R_swap_curve,T_fra_fit,interpolation_options)
    R_fra_init = [None]*len(T_fra)
    for i in range(0,len(T_fra)):
        R_fra_init[i] = R_fra_fit[value_in_list_returns_I_idx(T_fra[i],T_fra_fit)[1]]
    args = (T_fra,T_known,T_endo,T_fra_fit,R_fra_fit,fra_data,interpolation_options,scaling)
    result = minimize(zcb_curve_fra_fit_obj,R_fra_init,args = args,options={'disp': False})
    R_fra = list(result.x)
    R_endo = R_T_endo_from_R_T_fra(R_fra,T_fra,T_endo,fra_data)
    for i in range(0,len(T_fra_fit)):
        I_fra, idx_fra = value_in_list_returns_I_idx(T_fra_fit[i],T_fra)
        if I_fra is True:
            R_fra_fit[i] = R_fra[idx_fra]
        elif I_fra is False:
            I_endo, idx_endo = value_in_list_returns_I_idx(T_fra_fit[i],T_endo)
            if I_endo is True:
                R_fra_fit[i] = R_endo[idx_endo]
    return np.array(T_fra_fit), np.array(R_fra_fit)

def zcb_curve_interpolate(T,R,interpolation_options = {"method":"linear"},resolution = 1):
    T_inter = np.array([i*(1/(resolution*12)) for i in range(0,int(T[-1])*12*resolution + 1)])
    N = len(T_inter)
    p_inter = np.ones([N])
    R_inter = np.zeros([N])
    f_inter = np.zeros([N])
    if interpolation_options["method"] == "linear":
        for n in range(1,N):
            I_known, idx = value_in_list_returns_I_idx(T_inter[n],T)
            if I_known == True:
                R_inter[n] = R[idx]
            else:
                idx_before_x, idx_after_x = idx_before_after_in_list(T_inter[n],T)
                R_inter[n] = ((T[idx_after_x]-T_inter[n])/(T[idx_after_x]-T[idx_before_x]))*R[idx_before_x] + ((T_inter[n]-T[idx_before_x])/(T[idx_after_x]-T[idx_before_x]))*R[idx_after_x]
            p_inter[n] = np.exp(-R_inter[n]*T_inter[n])
        for n in range(1,N):
            if n < N-1:
                f_inter[n] = (np.log(p_inter[n-1])-np.log(p_inter[n+1]))/(T_inter[n+1]-T_inter[n-1])
            elif n == N - 1:
                f_inter[n] = (np.log(p_inter[n-1])-np.log(p_inter[n]))/(T_inter[n]-T_inter[n-1])
    elif interpolation_options["method"] == "hermite":
        indices = []
        for item in T:
            I_idx, idx = value_in_list_returns_I_idx(item,T_inter)
            if I_idx is True:
                indices.append(value_in_list_returns_I_idx(item,T_inter)[1])
        l, r = -int((interpolation_options["degree"]+1)/2),int(interpolation_options["degree"]/2)
        for i in range(-l,-r+len(T)):
            coef = hermfit(T[i+l:i+r+1],R[i+l:i+r+1],interpolation_options["degree"])
            coef_der = hermder(coef)
            for idx in range(indices[i+l],indices[i+r]+1):
                R_inter[idx] = hermval(T_inter[idx],coef)
                f_inter[idx] = hermval(T_inter[idx],coef_der)*T_inter[idx] + R_inter[idx]
                p_inter[idx] = np.exp(-R_inter[idx]*T_inter[idx])
    return p_inter, R_inter, f_inter, T_inter

def interpolate(x,y,x_inter,interpolation_options = {"method":"linear"}):
    N = len(x_inter)
    y_inter = [None]*N
    if interpolation_options["method"] == "linear":
        for n in range(0,N):
            I_known, idx = value_in_list_returns_I_idx(x_inter[n],x)
            if I_known == True:
                y_inter[n] = y[idx]
            else:
                idx_before_x, idx_after_x = idx_before_after_in_list(x_inter[n],x)
                y_inter[n] = ((x[idx_after_x]-x_inter[n])/(x[idx_after_x]-x[idx_before_x]))*y[idx_before_x] + ((x_inter[n]-x[idx_before_x])/(x[idx_after_x]-x[idx_before_x]))*y[idx_after_x]
    elif interpolation_options["method"] == "hermite":
        indices = []
        for item in x:
            I_idx, idx = value_in_list_returns_I_idx(item,x_inter)
            if I_idx is True:
                indices.append(value_in_list_returns_I_idx(item,x_inter)[1])
        l, r = -int((interpolation_options["degree"]+1)/2),int(interpolation_options["degree"]/2)
        for i in range(-l,-r+len(x)):
            coef = hermfit(x[i+l:i+r+1],y[i+l:i+r+1],interpolation_options["degree"])
            for idx in range(indices[i+l],indices[i+r]+1):
                y_inter[idx] = hermval(x_inter[idx],coef)
    return y_inter

def swap_indices(data,T):
    # Finding the swap indices
    for item in data:
        if item["instrument"] == "swap":
            indices = []
            if item["fixed_freq"] == "quarterly":
                for i in range(1,4*item["maturity"]+1):
                    indices.append(value_in_list_returns_I_idx(i*0.25,T)[1])
            elif item["fixed_freq"] == "semiannual":
                for i in range(1,2*item["maturity"]+1):
                    indices.append(value_in_list_returns_I_idx(i*0.5,T)[1])
            elif item["fixed_freq"] == "annual":
                for i in range(1,item["maturity"]+1):
                    indices.append(value_in_list_returns_I_idx(i,T)[1])
            item["indices"] = indices
    return data

def R_T_endo_from_R_T_fra(R_fra,T_fra,T_endo,fra_data):
    R_fra.reverse(), T_fra.reverse()
    R_endo = [None]*len(T_endo)
    for i in range(0,len(T_fra)):
        I_fra, dict_fra = value_in_list_of_dict_returns_I_idx(T_fra[i],fra_data,"maturity")
        if I_fra is True:
            idx_endo = value_in_list_returns_I_idx(dict_fra["exercise"],T_endo)[1]
            R_endo[idx_endo] = (R_fra[i]*T_fra[i] - np.log(1+(dict_fra["maturity"]-dict_fra["exercise"])*dict_fra["rate"]))/T_endo[idx_endo]
    R_fra.reverse(), T_fra.reverse()
    R_endo.reverse(), T_endo.reverse()
    for i in range(0,len(T_endo)):
        I_fra, dict_fra = value_in_list_of_dict_returns_I_idx(T_endo[i],fra_data,"maturity")
        if I_fra is True:
            idx_endo = value_in_list_returns_I_idx(dict_fra["exercise"],T_endo)[1]
            R_endo[idx_endo] = (R_endo[i]*T_endo[i] - np.log(1+(dict_fra["maturity"]-dict_fra["exercise"])*dict_fra["rate"]))/T_endo[idx_endo]
    R_endo.reverse(), T_endo.reverse()
    return R_endo

def zcb_curve_fra_fit_obj(R_fra,T_fra,T_known,T_endo,T_all,R_all,fra_data,interpolation_options,scaling = 1):
    sse = 0
    R_fra = list(R_fra)
    R_endo = R_T_endo_from_R_T_fra(R_fra,T_fra,T_endo,fra_data)
    for i in range(0,len(T_fra)):
        if T_fra[i] > min(T_known):
            sse += (R_all[value_in_list_returns_I_idx(T_fra[i],T_all)[1]] - R_fra[i])**2
    for i in range(0,len(T_endo)):
        if T_endo[i] > min(T_known):
            sse += (R_all[value_in_list_returns_I_idx(T_endo[i],T_all)[1]] - R_endo[i])**2
    sse *= scaling
    return sse

def zcb_curve_swap_fit_obj(R_knot,T_known,T_knot,T_all,R_known,swap_data,interpolation_options,scaling = 1):
    sse = 0
    R_knot = list(R_knot)
    R_all = interpolate(T_known + T_knot,R_known + R_knot,T_all,interpolation_options)
    p = zcb_prices_from_spot_rates(T_all,R_all)
    for n, swap in enumerate(swap_data):
        if swap["fixed_freq"] == "quarterly":
            alpha = 0.25
        if swap["fixed_freq"] == "semiannual":
            alpha = 0.5
        if swap["fixed_freq"] == "annual":
            alpha = 1
        S_swap = 0
        for idx in swap["indices"]:
            S_swap += alpha*p[idx]
        R_swap = (1 - p[swap["indices"][-1]])/S_swap
        sse += (R_swap - swap["rate"])**2
    sse *= scaling
    return sse

def value_in_list_returns_I_idx(value,list):
    output = False, None
    for i, item in enumerate(list):
        if abs(value-item) < 1e-6:
            output = True, i
            break
    return output

def idx_before_after_in_list(value,list):
    idx_before, idx_after = None, None
    if value < list[0]:
        idx_before, idx_after = 0, 1
    elif list[-1] < value:
        idx_before, idx_after = len(list)-2, len(list) - 1
    else:
        for i in range(0,len(list)):
            if list[i] < value:
                idx_before = i
            elif list[i] > value:
                idx_after = i
                break
    return idx_before, idx_after

def value_in_list_of_dict_returns_I_idx(value,L,name):
    output = False, None
    for item in L:
        if abs(value-item[name]) < 1e-6:
            output = True, item
            break
    return output
