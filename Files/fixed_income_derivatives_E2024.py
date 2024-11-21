import numpy as np
from scipy.stats import norm, ncx2
from scipy.optimize import minimize
from scipy.special import ndtr, gammainc
from numpy.polynomial.hermite import hermfit, hermval, hermder
import copy

# Conversions between ZCB prices, spot rates forward rates and libor rates
def zcb_prices_from_spot_rates(T,R):
    M = len(T)
    p = np.zeros([M])
    for i in range(0,M):
        if T[i] < 1e-8:
            p[i] = 1
        else:
            p[i] = np.exp(-R[i]*T[i])
    return p

def spot_rates_from_zcb_prices(T,p):
    M = len(T)
    r = np.zeros([M])
    for i in range(0,M):
        if T[i] < 1e-12:
            r[i] = np.nan
        else:
            r[i] = -np.log(p[i])/T[i]
    return r

def forward_rates_from_zcb_prices(T,p,horizon = 1):
    # horizon = 0 corresponds to approximated instantaneous forward rates
    M = len(T)
    f = np.zeros([M])
    if horizon == 0:
        f[0] = (np.log(p[0])-np.log(p[1]))/(T[1]-T[0])
        f[-1] = (np.log(p[-2])-np.log(p[-1]))/(T[-1]-T[-2])
        m = 1
        while m < M - 1.5:
            f[m] = (np.log(p[m-1])-np.log(p[m+1]))/(T[m+1]-T[m-1])
            m += 1
    elif 0 < horizon:
        m = horizon
        while m < M - 0.5:
            f[m] = (np.log(p[m-horizon])-np.log(p[m]))/(T[m]-T[m-horizon])
            m += 1
    return f

def forward_libor_rates_from_zcb_prices(T,p,horizon = 1):
    M = len(T)
    f = np.zeros([M])
    i = horizon
    while i < M - 0.5:
        f[i] = (p[i-horizon]-p[i])/(p[i]*(T[i]-T[i-horizon]))
        i += 1
    return f

def accrual_factor_from_zcb_prices(t,T_n,T_N,fixed_freq,T,p):
    T_fix = []
    if type(fixed_freq) == str:
        if fixed_freq == "quarterly":
            for i in range(1,int((T_N-T_n)*4) + 1):
                if T_n + i*0.25 > t:
                    T_fix.append(T_n + i*0.25)
        elif fixed_freq == "semiannual":
            for i in range(1,int((T_N-T_n)*2) + 1):
                if T_n + i*0.5 > t:
                    T_fix.append(T_n + i*0.5)
        elif fixed_freq == "annual":
            for i in range(1,int(T_N-T_n) + 1):
                if T_n + i > t:
                    T_fix.append(T_n + i)
    elif type(fixed_freq) == int or type(fixed_freq) == float or type(fixed_freq) == np.int32 or type(fixed_freq) == np.float64:
        for i in range(1,int((T_N-T_n)/fixed_freq) + 1):
            if T_n + i*fixed_freq > t:
                T_fix.append(T_n + i*fixed_freq)
    p_fix = np.array(for_values_in_list_find_value_return_value(T_fix,T,p))
    T_fix = np.array(T_fix)
    S = (T_fix[0] - t)*p_fix[0]
    for i in range(1,len(T_fix)):
        S += (T_fix[i] - T_fix[i-1])*p_fix[i]
    return S

def swap_rate_from_zcb_prices(t,T_n,T_N,fixed_freq,T,p,float_freq = 0,L = 0):
    S = accrual_factor_from_zcb_prices(t,T_n,T_N,fixed_freq,T,p)
    if t <= T_n:
        [p_n,p_N] = for_values_in_list_find_value_return_value([T_n,T_N],T,p)
        R = (p_n-p_N)/S
    elif t > T_n:
        if float_freq == 0:
            print(f"WARNING! Since t is after inception, 'float_freq' must be given as an argument")
            R = np.nan
        else:
            if type(float_freq) == str:
                if float_freq == "quarterly":
                    float_freq = 0.25
                elif float_freq == "semiannual":
                    float_freq = 0.5
                elif fixed_freq == "annual":
                    float_freq = 1
            i, I_done = 0, False
            while I_done == False and i*float_freq < T_N:
                if i*float_freq >= t:
                    T_n = i*float_freq
                    I_done = True
                i += 1
            if I_done == True:
                [p_n,p_N] = for_values_in_list_find_value_return_value([T_n,T_N],T,p)
                R = (((T_n-t)*L+1)*p_n-p_N)/S
            else:
                print(f"WARNING! Not able to compute the par swap rate")
                R = np.nan
    return R, S

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

# Cox-Ingersoll-Ross short rate sigma_model
def zcb_price_cir(r0,a,b,sigma,T):
    if type(T) == int or type(T) == float or type(T) == np.float32 or type(T) == np.float64:
        gamma = np.sqrt(a**2+2*sigma**2)
        D = (gamma+a)*(np.exp(gamma*T)-1)+2*gamma
        A = ((2*gamma*np.exp(0.5*T*(a+gamma)))/D)**((2*a*b)/(sigma**2))
        B = 2*(np.exp(gamma*T)-1)/D
        p = A*np.exp(-r0*B)
    elif type(T) == tuple or type(T) == list or type(T) == np.ndarray:
        M = len(T)
        p = np.zeros([M])
        for i in range(0,M):
            gamma = np.sqrt(a**2+2*sigma**2)
            D = (gamma+a)*(np.exp(gamma*T[i])-1)+2*gamma
            A = ((2*gamma*np.exp(0.5*T[i]*(a+gamma)))/D)**((2*a*b)/(sigma**2))
            B = 2*(np.exp(gamma*T[i])-1)/D
            p[i] = A*np.exp(-r0*B)
    else:
        print(f"T not a recognized type")
        p = False
    return p

def spot_rate_cir(r0,a,b,sigma,T):
    if type(T) == int or type(T) == float or type(T) == np.float64:
        if T < 1e-6:
            r = r0
        else:
            gamma = np.sqrt(a**2+2*sigma**2)
            D = (gamma+a)*(np.exp(gamma*T)-1)+2*gamma
            A = ((2*gamma*np.exp(0.5*T*(a+gamma)))/D)**((2*a*b)/(sigma**2))
            B = 2*(np.exp(gamma*T)-1)/D
            r = (-np.log(A)+r0*B)/(T)
    elif type(T) == tuple or type(T) == list or type(T) == np.ndarray:
        M = len(T)
        r = np.zeros([M])
        for i in range(0,M):
            if T[i] < 1e-6:
                r[i] = r0
            else:
                gamma = np.sqrt(a**2+2*sigma**2)
                D = (gamma+a)*(np.exp(gamma*T[i])-1)+2*gamma
                A = ((2*gamma*np.exp(0.5*T[i]*(a+gamma)))/D)**((2*a*b)/(sigma**2))
                B = 2*(np.exp(gamma*T[i])-1)/D
                r[i] = (-np.log(A)+r0*B)/(T[i])
    else:
        print(f"T not a recognized type")
        r = False
    return r

def forward_rate_cir(r0,a,b,sigma,T):
    if type(T) == int or type(T) == float or type(T) == np.float64:
        if T < 1e-6:
            f = r0
        else:
            c = (2*a*b)/(sigma**2)
            gamma = np.sqrt(a**2+2*sigma**2)
            N = 2*gamma*np.exp(0.5*T*(a+gamma))
            N_T = gamma*(gamma+a)*np.exp(0.5*T*(a+gamma))
            D = (gamma+a)*(np.exp(gamma*T)-1)+2*gamma
            D_T = gamma*(a+gamma)*np.exp(gamma*T)
            M = 2*(np.exp(gamma*T)-1)
            M_T = 2*gamma*np.exp(gamma*T)
            f = c*(-N_T/N+D_T/D)+r0*(M_T*D-M*D_T)/D**2
    elif type(T) == tuple or type(T) == list or type(T) == np.ndarray:
        N = len(T)
        f = np.zeros([N])
        for i in range(0,N):
            if T[i] < 1e-6:
                f[i] = r0
            else:
                c = (2*a*b)/(sigma**2)
                gamma = np.sqrt(a**2+2*sigma**2)
                N = 2*gamma*np.exp(0.5*T[i]*(a+gamma))
                N_T = gamma*(gamma+a)*np.exp(0.5*T[i]*(a+gamma))
                D = (gamma+a)*(np.exp(gamma*T[i])-1)+2*gamma
                D_T = gamma*(a+gamma)*np.exp(gamma*T[i])
                M = 2*(np.exp(gamma*T[i])-1)
                M_T = 2*gamma*np.exp(gamma*T[i])
                f[i] = c*(-N_T/N+D_T/D)+r0*(M_T*D-M*D_T)/D**2
    else:
        print(f"T not a recognized type")
        f = False
    return f

def mean_cir(r0,a,b,sigma,T):
    if T == np.inf:
        mean = b
    else:
        df = (4*a*b)/sigma**2
        nc = (4*a*np.exp(-a*T)*r0)/(sigma**2*(1-np.exp(-a*T)))
        factor = (sigma**2*(1-np.exp(-a*T)))/(4*a)
        mean = factor*ncx2.stats(df,nc,moments='m')
    return mean

def stdev_cir(r0,a,b,sigma,T):
    if T == np.inf:
        stdev = sigma*np.sqrt(b/(2*a))
    else:
        df = (4*a*b)/sigma**2
        nc = (4*a*np.exp(-a*T)*r0)/(sigma**2*(1-np.exp(-a*T)))
        factor = (sigma**2*(1-np.exp(-a*T)))/(4*a)
        stdev = factor*np.sqrt(ncx2.stats(df,nc,moments='v'))
    return stdev

def ci_cir(r0,a,b,sigma,T,size_ci,type_ci = "two_sided"):
    if type(T) == int or type(T) == float or type(T) == np.float64:
        if type_ci == "lower":
            if T < 1e-6:
                lb, ub = np.nan, np.nan
            else:
                df = (4*a*b)/sigma**2
                nc = (4*a*np.exp(-a*T)*r0)/(sigma**2*(1-np.exp(-a*T)))
                scaling = (4*a)/(sigma**2*(1-np.exp(-a*T)))
                lb, ub = ncx2.ppf(1-size_ci,df,nc)/scaling, np.inf
        elif type_ci == "upper":
            if T < 1e-6:
                lb, ub = np.nan, np.nan
            else:
                df = (4*a*b)/sigma**2
                nc = (4*a*np.exp(-a*T)*r0)/(sigma**2*(1-np.exp(-a*T)))
                scaling = (4*a)/(sigma**2*(1-np.exp(-a*T)))
                lb, ub = 0, ncx2.ppf(size_ci,df,nc)/scaling
        elif type_ci == "two_sided":
            if T < 1e-6:
                lb, ub = np.nan, np.nan
            else:
                df = (4*a*b)/sigma**2
                nc = (4*a*np.exp(-a*T)*r0)/(sigma**2*(1-np.exp(-a*T)))
                scaling = (4*a)/(sigma**2*(1-np.exp(-a*T)))
                lb, ub = ncx2.ppf((1-size_ci)/2,df,nc)/scaling, ncx2.ppf(size_ci+(1-size_ci)/2,df,nc)/scaling
    elif type(T) == tuple or type(T) == list or type(T) == np.ndarray:
        N = len(T)
        lb, ub = np.zeros([N]), np.zeros([N])
        if type_ci == "lower":
            for i in range(0,N):
                if T[i] < 1e-6:
                    lb[i], ub[i] = np.nan, np.nan

                else:
                    df = (4*a*b)/sigma**2
                    nc = (4*a*np.exp(-a*T[i])*r0)/(sigma**2*(1-np.exp(-a*T[i])))
                    scaling = (4*a)/(sigma**2*(1-np.exp(-a*T[i])))
                    lb[i], ub[i] = ncx2.ppf(1-size_ci,df,nc)/scaling, np.inf
        elif type_ci == "upper":
            for i in range(0,N):
                if T[i] < 1e-6:
                    lb[i], ub[i] = np.nan, np.nan
                else:
                    df = (4*a*b)/sigma**2
                    nc = (4*a*np.exp(-a*T[i])*r0)/(sigma**2*(1-np.exp(-a*T[i])))
                    scaling = (4*a)/(sigma**2*(1-np.exp(-a*T[i])))
                    lb[i], ub[i] = 0, ncx2.ppf(size_ci,df,nc)/scaling
        elif type_ci == "two_sided":
            for i in range(0,N):
                if T[i] < 1e-6:
                    lb[i], ub[i] = np.nan, np.nan
                else:
                    df = (4*a*b)/sigma**2
                    nc = (4*a*np.exp(-a*T[i])*r0)/(sigma**2*(1-np.exp(-a*T[i])))
                    scaling = (4*a)/(sigma**2*(1-np.exp(-a*T[i])))
                    lb[i], ub[i] = ncx2.ppf((1-size_ci)/2,df,nc)/scaling, ncx2.ppf(size_ci+(1-size_ci)/2,df,nc)/scaling
                    # (4*b)/(sigma**2*(1-np.exp(-b*tau[i])))*
    else:
        print(f"tau not a recognized type")
        lb,ub = False, False
    return lb, ub

def simul_cir(r0,a,b,sigma,M,T,method = "exact"):
    delta = T/M
    r = np.zeros([M+1])
    r[0] = r0
    if method == "exact":
        delta_sqrt = np.sqrt(delta)
        df = (4*a*b)/sigma**2
        factor = (sigma**2*(1-np.exp(-a*delta)))/(4*a)
        for m in range(1,M+1):
            nc = (4*a*np.exp(-a*delta)*r[m-1])/(sigma**2*(1-np.exp(-a*delta)))
            # r_hat = factor*np.random.noncentral_chisquare(df,nc)
            r_hat = factor*ncx2.rvs(df,nc)
            if r_hat > 0:
                r[m] = r_hat
            else:
                r[m] = r[m-1]
    elif method == "euler":
        delta_sqrt = np.sqrt(delta)
        Z = np.random.standard_normal(M)
        for m in range(1,M+1):
            r_hat = r[m-1] + a*(b-r[m-1])*delta + sigma*np.sqrt(r[m-1])*delta_sqrt*Z[m-1]
            if r_hat > 0:
                r[m] = r_hat
            else:
                r[m] = r[m-1]
    elif method == "milstein":
        delta_sqrt = np.sqrt(delta)
        Z = np.random.standard_normal(M)
        for m in range(1,M+1):
            r_hat = r[m-1] + a*(b-r[m-1])*delta + sigma*np.sqrt(r[m-1])*delta_sqrt*Z[m-1] + 0.5*sigma**2*delta*(Z[m-1]**2-1)
            if r_hat > 0:
                r[m] = r_hat
            else:
                r[m] = r[m-1]
    return r

# Vasicek short rate model
def zcb_price_vasicek(r0,a,b,sigma,T):
    if type(T) == int or type(T) == float or type(T) == np.int32 or type(T) == np.float64:
        B = (1/a)*(1-np.exp(-a*T))
        A = (B-T)*(a*b-0.5*sigma**2)/(a**2)-(sigma**2*B**2)/(4*a)
        p = np.exp(A-r0*B)
    elif type(T) == tuple or type(T) == list or type(T) == np.ndarray:
        M = len(T)
        p = np.zeros([M])
        for i in range(0,M):
            B = (1/a)*(1-np.exp(-a*T[i]))
            A = (B-T[i])*(a*b-0.5*sigma**2)/(a**2)-(sigma**2*B**2)/(4*a)
            p[i] = np.exp(A-r0*B)
    else:
        print(f"T not of a recognized type")
        p = False
    return p

def spot_rate_vasicek(r0,a,b,sigma,T):
    if type(T) == int or type(T) == float or type(T) == np.int32 or type(T) == np.float64:
        B = (1/a)*(1-np.exp(-a*T))
        A = (B-T)*(a*b-0.5*sigma**2)/(a**2)-(sigma**2*B)/(4*a)
        if T < 1e-6:
            r = r0
        elif T >= 1e-6:
            r = (-A+r0*B)/T
    elif type(T) == tuple or type(T) == list or type(T) == np.ndarray:
        M = len(T)
        r = np.zeros([M])
        for i in range(0,M):
            B = (1/a)*(1-np.exp(-a*T[i]))
            A = (B-T[i])*(a*b-0.5*sigma**2)/(a**2)-(sigma**2*B)/(4*a)
            if T[i] < 1e-6:
                r[i] = r0
            else:
                r[i] = (-A+r0*B)/T[i]
    else:
        print(f"T not of a recognized type")
        r = False
    return r

def forward_rate_vasicek(r0,a,b,sigma,T):
    if type(T) == int or type(T) == float or type(T) == np.int32 or type(T) == np.float64:
        B = (1/a)*(1-np.exp(-a*T))
        B_T = np.exp(-a*T)
        if T < 1e-6:
            f = r0
        elif T >= 1e-6:
            f = (1-B_T)*(a*b-0.5*sigma**2)/(a**2) + (sigma**2*B*B_T)/(2*a) + r0*B_T
    elif type(T) == tuple or type(T) == list or type(T) == np.ndarray:
        M = len(T)
        f = np.zeros([M])
        for i in range(0,M):
            B = (1/a)*(1-np.exp(-a*T[i]))
            B_T = np.exp(-a*T[i])
            if T[i] < 1e-6:
                f[i] = r0
            else:
                f[i] = (1-B_T)*(a*b-0.5*sigma**2)/(a**2) + (sigma**2*B*B_T)/(2*a) + r0*B_T
    else:
        print(f"T not of a recognized type")
        f = False
    return f

def mean_vasicek(r0,a,b,sigma,T):
    if T == np.inf:
        mean = b/a
    else:
        mean = r0*np.exp(-a*T) + b/a*(1-np.exp(-a*T))
    return mean

def stdev_vasicek(r0,a,b,sigma,T):
    if T == np.inf:
        std = np.sqrt(sigma**2/(2*a))
    else:
        std = np.sqrt(sigma**2/(2*a)*(1-np.exp(-2*a*T)))
    return std

def ci_vasicek(r0,a,b,sigma,T,size_ci,type_ci = "two_sided"):
    if type(T) == int or type(T) == float or type(T) == np.int32 or type(T) == np.float64:
        if type_ci == "lower":
            z = norm.ppf(size_ci,0,1)
            if T < 1e-10:
                lb, ub = np.nan, np.nan
            elif T == np.inf:
                mean = b/a
                std = np.sqrt(sigma**2/(2*a))
                lb, ub = mean - z*std, np.inf
            else:
                mean = r0*np.exp(-a*T) + b/a*(1-np.exp(-a*T))
                std = np.sqrt(sigma**2/(2*a)*(1-np.exp(-2*a*T)))
                lb, ub = mean - z*std, np.inf
        elif type_ci == "upper":
            z = norm.ppf(size_ci,0,1)
            if T < 1e-10:
                lb, ub = np.nan, np.nan
            elif T == np.inf:
                mean = b/a
                std = np.sqrt(sigma**2/(2*a))
                lb, ub = -np.inf, mean + z*std
            else:
                mean = r0*np.exp(-a*T) + b/a*(1-np.exp(-a*T))
                std = np.sqrt(sigma**2/(2*a)*(1-np.exp(-2*a*T)))
                lb, ub = -np.inf, mean + z*std
        elif type_ci == "two_sided":
            z = norm.ppf(size_ci + 0.5*(1-size_ci),0,1)
            if T < 1e-10:
                lb, ub = np.nan, np.nan
            elif T == np.inf:
                mean = b/a
                std = np.sqrt(sigma**2/(2*a))
                lb, ub = mean - z*std, mean + z*std
            else:
                mean = r0*np.exp(-a*T) + b/a*(1-np.exp(-a*T))
                std = np.sqrt(sigma**2/(2*a)*(1-np.exp(-2*a*T)))
                lb, ub = mean - z*std, mean + z*std
        print(f"type_ci: {type_ci}, z: {z}")
    elif type(T) == tuple or type(T) == list or type(T) == np.ndarray:
        N = len(T)
        lb, ub = np.zeros([N]), np.zeros([N])
        if type_ci == "lower":
            z = norm.ppf(size_ci,0,1)
            for i in range(0,N):
                if T[i] < 1e-10:
                    lb[i], ub[i] = np.nan, np.nan
                else:
                    mean = r0*np.exp(-a*T[i]) + b/a*(1-np.exp(-a*T[i]))
                    std = np.sqrt(sigma**2/(2*a)*(1-np.exp(-2*a*T[i])))
                    lb[i], ub[i] = mean - z*std, np.inf
        elif type_ci == "upper":
            z = norm.ppf(size_ci,0,1)
            for i in range(0,N):
                if T[i] < 1e-10:
                    lb[i], ub[i] = np.nan, np.nan
                else:
                    mean = r0*np.exp(-a*T[i]) + b/a*(1-np.exp(-a*T[i]))
                    std = np.sqrt(sigma**2/(2*a)*(1-np.exp(-2*a*T[i])))
                    lb[i], ub[i] = -np.inf, mean + z*std
        elif type_ci == "two_sided":
            z = norm.ppf(size_ci + 0.5*(1-size_ci),0,1)
            for i in range(0,N):
                if T[i] < 1e-10:
                    lb[i], ub[i] = np.nan, np.nan
                else:
                    mean = r0*np.exp(-a*T[i]) + b/a*(1-np.exp(-a*T[i]))
                    std = np.sqrt(sigma**2/(2*a)*(1-np.exp(-2*a*T[i])))
                    lb[i], ub[i] = mean - z*std, mean + z*std
    else:
        print(f"T is not of recognized type")
        lb,ub = False, False
    return lb, ub

def simul_vasicek(r0,a,b,sigma,M,T,method = "exact",seed = None):
    if seed is not None:
        np.random.seed(seed)
    delta = T/M
    r = np.zeros([M+1])
    r[0] = r0
    Z = np.random.standard_normal(M)
    if method == "exact":
        for m in range(1,M+1):
            r[m] = r[m-1]*np.exp(-a*delta) + (b/a)*(1-np.exp(-a*delta)) + sigma*np.sqrt((1-np.exp(-2*a*delta))/(2*a))*Z[m-1]
    elif method == "euler" or method == "milstein":
        delta_sqrt = np.sqrt(delta)
        for m in range(1,M+1):
            r[m] = r[m-1] + (b-a*r[m-1])*delta + sigma*delta_sqrt*Z[m-1]
    return r

def euro_option_price_vasicek(K,T1,T2,p_T1,p_T2,a,sigma,type = "call"):
    sigma_p = (sigma/a)*(1-np.exp(-a*(T2-T1)))*np.sqrt((1-np.exp(-2*a*T1))/(2*a))
    d1 = (np.log(p_T2/(p_T1*K)))/sigma_p + 0.5*sigma_p
    d2 = d1 - sigma_p
    if type == "call":
        price = p_T2*ndtr(d1) - p_T1*K*ndtr(d2)
    elif type == "put":
        price = p_T1*K*ndtr(-d2) - p_T2*ndtr(-d1)
    return price

def caplet_prices_vasicek(sigma,strike,a,T,p):
    price_caplet = np.zeros([len(T)])
    for i in range(2,len(T)):
        price_caplet[i] = (1 + (T[i]-T[i-1])*strike)*euro_option_price_vasicek(1/(1 + (T[i]-T[i-1])*strike),T[i-1],T[i],p[i-1],p[i],a,sigma,type = "put")
    return price_caplet

def fit_vasicek_obj(param,R_star,T,scaling = 1):
    r0, a, b, sigma = param
    M = len(T)
    R_fit = spot_rate_vasicek(r0,a,b,sigma,T)
    y = 0
    for m in range(0,M):
        y += scaling*(R_fit[m] - R_star[m])**2
    return y

def fit_vasicek_no_sigma_obj(param,sigma,R_star,T,scaling = 1):
    r0, a, b = param
    M = len(T)
    R_fit = spot_rate_vasicek(r0,a,b,sigma,T)
    y = 0
    for m in range(0,M):
        y += scaling*(R_fit[m] - R_star[m])**2
    return y

def fit_cir_obj(param,R_star,T,scaling = 1):
    r0, a, b, sigma = param
    M = len(T)
    R_fit = spot_rate_cir(r0,a,b,sigma,T)
    y = 0
    for m in range(0,M):
        y += scaling*(R_fit[m] - R_star[m])**2
    return y

def fit_cir_no_sigma_obj(param,sigma,R_star,T,scaling = 1):
    r0, a, b = param
    M = len(T)
    R_fit = spot_rate_cir(r0,a,b,sigma,T)
    y = 0
    for m in range(0,M):
        y += scaling*(R_fit[m] - R_star[m])**2
    return y

# Fitting the initial term structure of forward rates (For use in the Ho-Lee and Hull-White extended Vasicek models)
def theta(t,sigma,args):
    if args["method"] == "nelson-siegel":
        a = args["a"]
        b = args["b"]
        if type(t) == int or type(t) == float or type(t) == np.int32 or type(t) == np.float64:
            K = len(a)
            theta = -a[0]*b[0]*np.exp(-b[0]*t) + sigma**2*t
            for k in range(1,K):
                theta += a[k]*k*t**(k-1)*np.exp(-b[k]*t) - a[k]*b[k]*t**k*np.exp(-b[k]*t)
        elif type(t) == tuple or type(t) == list or type(t) == np.ndarray:
            K = len(a)
            M = len(t)
            theta = np.zeros([M])
            for m in range(0,M):
                theta[m] = -a[0]*b[0]*np.exp(-b[0]*t[m]) + sigma**2*t[m]
                for k in range(1,K):
                    theta[m] += a[k]*k*t[m]**(k-1)*np.exp(-b[k]*t[m]) - a[k]*b[k]*t[m]**k*np.exp(-b[k]*t[m])
    if args["method"] == "empirical":
        T = args["T"]
        f_star_T = args["f_star_T"]
        M, N = len(t), len(T)
        theta = np.zeros([M])
        i, j = 0, 0
        while i < M:
            while j < N:
                if t[i] < T[j]:
                    print(f"WARNING! Not able to compute theta for t: {t[i]}. t less than T, t: {t[i]}, T: {T[j]}")
                    i += 1
                elif T[j] <= t[i] <= T[j+1]:
                    w_right = (t[i] - T[j])/(T[j+1]-T[j])
                    theta[i] = w_right*f_star_T[j+1] + (1-w_right)*f_star_T[j]
                    if i + 1 > M - 1:
                        j = N
                    i += 1
                elif t[i] > T[j+1]:
                    if j + 1 > N - 1:
                        print(f"WARNING! Not able to compute theta for t: {t[i]}. t greater than T, t: {t[i]}, T: {T[j]}")
                    else:
                        j += 1
    return theta

# Ho-Lee model
def simul_ho_lee(r0,theta_args,sigma,N,M,T,method = "euler"):
    r = np.zeros([N,M+1])
    r[0] = r0
    delta = T/M
    Z = np.random.standard_normal(size=(N,M))
    if method == "euler":
        delta_sqrt = np.sqrt(delta)
        if theta_args["method"] == "nelson-siegel":
            t = np.array([m*delta for m in range(0,M+1)])
            theta_simul = theta(t,sigma,theta_args)
            for n in range(0,N):
                for m in range(1,M+1):
                    r[n,m] = r[n,m-1] + theta_simul[m]*delta + sigma*delta_sqrt*Z[n,m-1]
        elif theta_args["method"] == "empirical":
            t = np.array([i*delta for i in range(0,M+1)])
            theta_values = theta(t,sigma,theta_args)
            for n in range(0,N):
                for m in range(1,M+1):
                    r[n,m] = r[n,m-1] + theta_values[m]*delta + sigma*delta_sqrt*Z[n,m-1]
    return r

# Nelson-Siegel function
def F_ns(param,T):
    if type(T) == int or type(T) == float or type(T) == np.int32 or type(T) == np.float64:
        f_inf, a, b = param
        K = len(a)
        F = f_inf*T + a[0]*np.exp(-b[0]*T)
        for k in range(0,K):
            F += a[k]*b[k]**(-k-1)*gammainc(k+1,b[k]*T)
    elif type(T) == tuple or type(T) == list or type(T) == np.ndarray:
        f_inf, a, b = param
        K = len(a)
        M = len(T)
        F = np.zeros([M])
        for m in range(0,M):
            F[m] = f_inf*T[m]
            for k in range(0,K):
                F[m] += a[k]*b[k]**(-k-1)*gammainc(k+1,b[k]*T[m])
    return F

def f_ns(param,T):
    if type(T) == int or type(T) == float or type(T) == np.int32 or type(T) == np.float64:
        f_inf, a, b = param
        K = len(a)
        f = f_inf
        for k in range(0,K):
            f += a[k]*T**k*np.exp(-b[k]*T)
    elif type(T) == tuple or type(T) == list or type(T) == np.ndarray:
        f_inf, a, b = param
        K = len(a)
        M = len(T)
        f = np.zeros([M])
        for m in range(0,M):
            f[m] = f_inf
            for k in range(0,K):
                f[m] += a[k]*T[m]**k*np.exp(-b[k]*T[m])
    return f

def f_ns_jac(param,T):
    f_inf, a, b = param
    N = len(a) - 1
    jac = np.zeros([2*(N+1)+1])
    jac[0] = 1
    for n in range(0,N+1):
        jac[1+n] = T**n*np.exp(-b[n]*T)
        jac[1+N+1+n] = -a[n]*T**(n+1)*np.exp(-b[n]*T)
    return jac

def f_ns_hess(param,T):
    f_inf, a, b = param
    N = len(a) - 1
    hess = np.zeros([2*(N+1)+1,2*(N+1)+1])
    for n in range(0,N+1):
        hess[1+n,1+N+1+n] = - T**(n+1)*np.exp(-b[n]*T)
        hess[1+N+1+n,1+n] = - T**(n+1)*np.exp(-b[n]*T)
        hess[1+N+1+n,1+N+1+n] = a[n]*T**(n+2)*np.exp(-b[n]*T)
    return hess

def f_ns_T(param,T):
    if type(T) == int or type(T) == float or type(T) == np.float32 or type(T) == np.float64:
        a, b = param
        N = len(a)
        f_T = -a[0]*b[0]*np.exp(-b[0]*T)
        for n in range(1,N):
            f_T += a[n]*n*T**(n-1)*np.exp(-b[n]*T) - a[n]*b[n]*T**n*np.exp(-b[n]*T)
    elif type(T) == tuple or type(T) == list or type(T) == np.ndarray:
        a, b = param
        N = len(a)
        M = len(T)
        f_T = np.zeros([M])
        for m in range(0,M):
            f_T[m] = -a[0]*b[0]*np.exp(-b[0]*T[m])
            for n in range(1,N):
                f_T[m] += a[n]*n*T[m]**(n-1)*np.exp(-b[n]*T[m]) - a[n]*b[n]*T[m]**n*np.exp(-b[n]*T[m])
    return f_T

def theta_ns(param,t):
    if type(t) == int or type(t) == float or type(t) == np.float32 or type(t) == np.float64:
        f_inf, a, b, sigma = param
        print(f_inf, a, b, sigma)
        K = len(a)
        theta = -a[0]*b[0]*np.exp(-b[0]*t) + sigma**2*t
        for k in range(1,K):
            theta += a[k]*k*t**(k-1)*np.exp(-b[k]*t) - a[k]*b[k]*t**k*np.exp(-b[k]*t)
    elif type(t) == tuple or type(t) == list or type(t) == np.ndarray:
        f_inf, a, b, sigma = param
        K = len(a)
        M = len(t)
        theta = np.zeros([M])
        for m in range(0,M):
            theta[m] = -a[0]*b[0]*np.exp(-b[0]*t[m]) + sigma**2*t[m]
            for k in range(1,K):
                theta[m] += a[k]*k*t[m]**(k-1)*np.exp(-b[k]*t[m]) - a[k]*b[k]*t[m]**k*np.exp(-b[k]*t[m])
    return theta

# Caplets
def black_caplet_price(sigma,T,R,alpha,p,L,type = "call"):
    d1 = (np.log(L/R) + 0.5*sigma**2*(T-alpha))/(sigma*np.sqrt(T-alpha))
    d2 = (np.log(L/R) - 0.5*sigma**2*(T-alpha))/(sigma*np.sqrt(T-alpha))
    if type == 'put':
        price = alpha*p*(R*ndtr(-d2) - L*ndtr(-d1))
    else:
        price = alpha*p*(L*ndtr(d1) - R*ndtr(d2))
    return price

def black_caplet_delta(sigma,T,R,alpha,p,L,type = "call"):
    d1 = (np.log(L/R) + 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    d2 = (np.log(L/R) - 0.5*sigma**2*(T-alpha))/(sigma*np.sqrt(T-alpha))
    if type == "call":
        # p_prev = p*(1+alpha*L)
        price = alpha*p*(L*ndtr(d1) - R*ndtr(d2))
        delta = alpha*p*ndtr(d1) - alpha/(1+alpha*L)*price
    return delta

def black_caplet_gamma(sigma,T,R,alpha,p,L,type = "call"):
    d1 = (np.log(L/R) + 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    d2 = (np.log(L/R) - 0.5*sigma**2*(T-alpha))/(sigma*np.sqrt(T-alpha))
    if type == "call":
        gamma = alpha*p*(norm.pdf(d1)/(L*sigma*np.sqrt(T-alpha))-(2*alpha)/((1+alpha*L)**2)*(alpha*R*ndtr(d2) + ndtr(d1)))
    return gamma

def black_caplet_vega(sigma,T,R,alpha,p,L,type = "call"):
    if type == "call":
        d1 = (np.log(L/R) + 0.5*sigma**2*T)/(sigma*np.sqrt(T))
        vega = alpha*p*L*norm.pdf(d1) * np.sqrt(T)
    return vega

def black_caplet_theta(sigma,T,r,R,alpha,p,L,type = "call"):
    d1 = (np.log(L/R) + 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    d2 = (np.log(L/R) - 0.5*sigma**2*(T-alpha))/(sigma*np.sqrt(T-alpha))
    price = alpha*p*(L*ndtr(d1) - R*ndtr(d2))
    if type == "call":
        # p_prev = p*(1+alpha*L)
        price = alpha*p*(L*ndtr(d1) - R*ndtr(d2))
        theta = r*price - alpha*p*(sigma*L*norm.pdf(d1))/(2*np.sqrt(T))
    return theta

def black_caplet_iv(C,T,R,alpha,p,L, iv0 = 0.2, max_iter = 200, prec = 1.0e-5):
    iv = iv0
    for i in range(0,max_iter):
        price = black_caplet_price(iv,T,R,alpha,p,L,type = "call")
        vega = black_caplet_vega(iv,T,R,alpha,p,L,type = "call")
        diff = C - price
        if abs(diff) < prec:
            return iv
        iv += diff/vega
    return iv

# Swatiopns
def black_swaption_price(sigma,T,K,S,R,type = "call"):
    d1 = (np.log(R/K) + 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    d2 = (np.log(R/K) - 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    if type == 'put':
        price = S*(K*ndtr(-d2) - R*ndtr(-d1))
    else:
        price = S*(R*ndtr(d1) - K*ndtr(d2))
    return price

def black_swaption_delta(sigma,T,K,S,R,type = "call"):
    d1 = (np.log(R/K) + 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    d2 = (np.log(R/K) - 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    if type == 'call':
        price = S*(R*ndtr(d1) - K*ndtr(d2))
        delta = (S/R)*ndtr(d1) - price/R
    return delta

def black_swaption_gamma(sigma,T,K,S,R,type = "call"):
    d1 = (np.log(R/K) + 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    d2 = (np.log(R/K) - 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    if type == 'call':
        price = S*(R*ndtr(d1) - K*ndtr(d2))
        gamma = (2/R**2)*price + (S/R)*(norm.pdf(d1)/(sigma*np.sqrt(T)) - 2*ndtr(d1))
    return gamma

def black_swaption_vega(sigma,T,K,S,R,type = "call"):
    d1 = (np.log(R/K) + 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    if type == 'call':
        vega = S*R*norm.pdf(d1) * np.sqrt(T)
    return vega

def black_swaption_theta(sigma,T,K,S,r,R,type = "call"):
    d1 = (np.log(R/K) + 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    d2 = (np.log(R/K) - 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    if type == 'call':
        price = S*(R*ndtr(d1) - K*ndtr(d2))
        theta = r*price - S*R*sigma*norm.pdf(d1)/(2*np.sqrt(T))
    return theta

def black_swaption_iv(C,T,K,S,R,type = "call", iv0 = 0.2, max_iter = 1000, prec = 1.0e-10):
    iv = iv0
    for i in range(0,max_iter):
        price = black_swaption_price(iv,T,K,S,R,type = "call")
        vega = black_swaption_vega(iv,T,K,S,R,type = "call")
        diff = C - price
        if abs(diff) < prec:
            return iv
        iv += diff/vega
    return iv

# SABR model
def sigma_sabr(K,T,F_0,sigma_0,beta,upsilon,rho,type = "call"):
    if abs(F_0-K) < 1e-8:    # SABR ATM formula
        sigma = sigma_0*F_0**(beta-1)*(1+(((1-beta)**2/24)*(sigma_0**2*(F_0)**(2*beta-2)) + (rho*beta*upsilon*sigma_0/4)*(F_0)**(beta-1) + (2-3*rho**2)/24*upsilon**2)*T)
    else:
        z = (upsilon/sigma_0)*(F_0*K)**((1-beta)/2)*np.log(F_0/K)
        x = np.log((np.sqrt(1-2*rho*z+z**2) + z - rho)/(1-rho))
        D = (F_0*K)**((1-beta)/(2))*(1 + ((1-beta)**2/24)*np.log2(F_0/K) + ((1-beta)**4/1920)*np.emath.logn(4,F_0/K))
        # print(z,x,D)
        A = 1 + (((1-beta)**2/24)*sigma_0**2*(F_0*K)**(beta-1) + (rho*beta*upsilon*sigma_0/4)*(F_0*K)**((beta-1)/2) + ((2-3*rho**2)/24)*upsilon**2)*T
        sigma = (sigma_0/D)*(z/x)*A
    return sigma

def sabr_simul(F_0,sigma_0,beta,upsilon,rho,M,T):
    sigma, F = np.zeros([M+1]), np.zeros([M+1])
    sigma[0], F[0] = sigma_0, F_0
    delta = T/M
    Z = np.random.standard_normal([2,M])
    # print(np.average(Z[1,:]),np.var(Z[1,:]))
    delta_sqrt = np.sqrt(delta)
    rho_sqrt = np.sqrt(1-rho**2)
    for m in range(1,M+1):
        F[m] = F[m-1] + sigma[m-1]*F[m-1]**beta*delta_sqrt*Z[0,m-1]
        # if F[m] < 0:
        #     F[m] = F[m-1]
        sigma[m] = sigma[m-1] + upsilon*sigma[m-1]*delta_sqrt*(rho*Z[0,m-1] + rho_sqrt*Z[1,m-1])
        # if sigma[m] < 0:
        #     sigma[m] = sigma[m-1]
    return F, sigma

def fit_sabr_obj(param,sigma_market,K,T,R):
    sigma_0, beta, upsilon, rho = param
    N = len(sigma_market)
    sse = 0
    for n in range(0,N):
        sigma_model = sigma_sabr(K[n],T,R,sigma_0,beta,upsilon,rho,type = "call")
        sse += (sigma_market[n]-sigma_model)**2
    return sse

# Statistics
def data_into_bins(data,N_bins,bin_min = "default",bin_max = "default"):
    # Divide the data in a one-simensional np.array into N_bins bins of equal size.
    N = len(data)
    data = np.sort(data,kind = "stable")
    if bin_min == "default":
        bin_min = data[0]
    if bin_max == "default":
        bin_max = data[-1]
    limits_bins = np.array([bin_min + i*(bin_max-bin_min)/N_bins for i in range(0,N_bins+1)])
    freq = np.zeros([N_bins])
    data_bins = []
    i, j = 0, 0
    while i < N_bins:
        freq_temp = 0
        data_temp = []
        while data[j] <= limits_bins[i+1]:
            data_temp.append(data[j])
            freq_temp += 1
            if j < N - 1:
                j += 1
            elif j == N - 1:
                break
        data_bins.append(np.array(data_temp))
        freq[i] = freq_temp
        i += 1
    return np.array(data_bins), limits_bins, freq

# List operations
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
    # Fitting the swap portion of the curve
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
