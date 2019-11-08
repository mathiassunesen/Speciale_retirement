# global modules
import numpy as np
from numba import njit, prange


##################################
####      index functions    #####
##################################
@njit(parallel=True)
def age(t,par): 
    """ convert model time (t) to real age (The inverse of inv_age function) """
    return t+par.start_T

@njit(parallel=True)
def inv_age(age,par): 
    """ convert age to model time (t) (The inverse of age function) """
    return age-par.start_T

@njit(parallel=True)
def ra_look_up(t,st,ra,d,par):
    """ look up the right retirement status (ra) in period t conditional on labor market status and current ra"""

    # if not eligible to erp status is always ra=2
    if state_translate(st,'elig',par) == 0:
        ra_look = 2

    # if retired ra is equal to current ra or 0
    elif d == 0:
        if t >= par.T_oap-1:
            ra_look = 0
        else:
            ra_look = ra

    # if working ra is equal to ra if retiring next period
    elif d == 1:
        if t+1 >= par.T_two_year:
            ra_look = 0
        elif t+1 >= par.T_erp:
            ra_look = 1
        else:
            ra_look = 2

    return ra_look

@njit(parallel=True)
def d_plus(t,d_t,par):
    """ find labor choice set for next period (t+1) """
    
    # if retired: only choice is to stay retired
    if d_t == 0:
        d_plus = np.array([0])

    # working
    elif d_t == 1:
        
        # if working but forced to retire: only choice is to stay retired
        if t+1 == par.Tr-1:
            d_plus = np.array([0])
        
        # if working and not forced to retire: full choice set
        elif t+1 < par.Tr-1:
            d_plus = np.array([0,1])

    return d_plus  

@njit(parallel=True)
def d_plus_int(t,d_t,par):
    """ the same a d_plus function, but returns the highest integer in the choice set """
    
    if d_t == 0:
        d_plus = 0
    elif d_t == 1:
        if t+1 >= par.Tr-1:
            d_plus = 0
        elif t+1 < par.Tr-1:
            d_plus = 1

    return d_plus  

@njit(parallel=True)
def d_plus_c(t,ad,d_h,d_w,par):
    """ returns the next period (t+1) choice set for couples """

    # if both retired: both stay retired
    if d_h == 0 and d_w == 0:
        d_plus = np.array([0])

    # only wife is working
    elif d_h == 0 and d_w == 1:

        # if wife is forced to retire: both stay retired
        if t+1+ad == par.Tr-1:
            d_plus = np.array([0])

        # wife is not forced to retire: husband stays retired, but wife has full choice set
        else:
            d_plus = np.array([0,1])

    # only husband working
    elif d_h == 1 and d_w == 0:

        # if husband forced to retire: both stay retired
        if t+1 == par.Tr-1:
            d_plus = np.array([0])

        # if husband not forced to retire: wife stays retired, but husband has full choice set
        else:
            d_plus = np.array([0,2])

    # both working
    elif d_h == 1 and d_w == 1:

        # both forced to retire next period (same age)
        if ad == 0 and t+1 == par.Tr-1:     
            d_plus = np.array([0])
        
        # husband forced to retire next period (but not wife)
        elif ad < 0 and t+1 == par.Tr-1:    
            d_plus = np.array([0,1])

        # wife forced to retire next period (but not husband)
        elif ad > 0 and t+1+ad == par.Tr-1: 
            d_plus = np.array([0,2])
        
        # none are forced to retire next period        
        else:                               
            d_plus = np.array([0,1,2,3])

    return d_plus          

@njit(parallel=True)
def d_c(d_h,d_w):
    
    if d_h == 0 and d_w == 0:
        d = 0
    elif d_h == 0 and d_w == 1:
        d = 1
    elif d_h == 1 and d_w == 0:
        d = 2
    elif d_h == 1 and d_w == 1:
        d = 3
    return d

@njit(parallel=True)
def state_translate(st,ind,par):
    """ translate st to info on whether ind is true. ind can be elig or high_skilled """

    ST = par.ST[st]
    if ind == 'elig':
        if ST[0] == 1:
            return 1
        else:
            return 0
    elif ind == 'high_skilled':
        if ST[1] == 1:
            return 1
        else:
            return 0

##################################
####      precompute          ####
##################################
def precompute(model):
    """ precompute income streams and survival (wrapper) """
    if model.couple:
        pass

    else:
        precompute_inc_single(model.par)
        precompute_survival(model.par)


def precompute_survival(par):
    """ precompute survival probabilities """
    
    # unpack    
    T = par.T+par.ad_min+par.ad_max
    NMA = len(par.MA)
    NST = len(par.ST)
    
    # initialize
    par.survival = np.nan*np.zeros((T,NMA,NST))
    
    # compute
    for t in range(T):
        for ma in range(NMA):
            for st in range(NST):
                par.survival[t,ma,st] = survival(t-par.ad_min,ma,st,par)

def precompute_inc_single(par):
    """ precompute income streams for singles """

    # states
    ad_min = par.ad_min
    ad_max = par.ad_max
    NMA = len(par.MA)
    NST = len(par.ST)
    RA = np.array([0,1,2])
    NRA = len(RA)

    # shocks (labor income)
    xi = par.xi    
    Nxi = par.Nxi    

    # time lines
    extend = ad_min+ad_max
    T = par.T+extend                    # total time
    Tr = par.Tr+extend                  # for labor income
    T_erp = par.T_oap-par.T_erp+extend  # for erp
    T_oap = par.T-par.T_oap+extend      # for oap

    # initialize
    par.oap = np.nan*np.zeros(T_oap)
    par.labor = np.nan*np.zeros((Tr,NMA,NST,Nxi)) 
    par.erp = np.nan*np.zeros((T_erp,NMA,NST,NRA))
        
    # precompute income
    for t in range(T):

        # oap
        if t-ad_min >= par.T_oap:
            pre = oap_pretax(t-ad_min,par)
            par.oap[:] = posttax(t-ad_min,pre,par,labor=False)

        for ma in range(NMA):
            for st in range(NST):

                # labor income
                if t-ad_min < par.Tr:
                    pre = labor_pretax(t-ad_min,ma,st,par)
                    par.labor[t,ma,st] = posttax(t-ad_min,pre,par,labor=True,shock=xi[ma])

                # erp
                if par.T_erp <= t-ad_min < par.T_oap:
                    for ra in range(NRA):
                        pre = erp_pretax(t-ad_min,ma,st,ra,par)
                        par.erp[t-par.T_erp,ma,st,ra] = posttax(t-ad_min,pre,par,labor=False)

##################################
####         look up         #####
##################################
@njit(parallel=True)
def inc_lookup_single(d,t,ma,st,ra,par):
    """ look up in the precomputed income streams for singles (posttax)"""
    
    # retired (returns a float so convert to array before returning)
    if d == 0:

        # erp
        if par.T_erp <= t < par.T_oap:
            return np.array([par.erp[t-par.T_erp,ma,st,ra]]) # subtract par.T_erp to get right time index

        # oap
        elif t >= par.T_oap:
            return np.array([par.oap[t-par.T_oap]]) # subtract par.T_oap to get right time index

        # no pension
        else:
            return np.array([0.0])
    
    # working
    elif d == 1:
        return par.labor[t,ma,st]   # don't need to subtract here since time is already aligned

@njit(parallel=True)
def survival_lookup_single(t,ma,st,par):
    """ look up in the precomputed survival probabilities for singles """
    return par.survival[t,ma,st]

##################################
####        tax system       #####
##################################
@njit(parallel=True)
def posttax(t,inc_pre,par,labor,shock=np.array([1.0]),spouse_inc=np.nan):
    """ compute posttax income potentially including shocks """    

    # shocks and labor market contribution is only applied to labor income
    inc = inc_pre*shock
    personal_income = (1 - par.tau_LMC*labor)*inc

    # extra deduction (fradrag) to use in policy simulation
    if labor and age(t,par) >= par.oap_age:
        taxable_income = personal_income - np.minimum(par.WD*inc,par.WD_upper) - par.fradrag
    else:
        taxable_income = personal_income - np.minimum(par.WD*inc,par.WD_upper)

    # potential shared spouse deduction
    if par.couple:
        y_low_l = par.y_low + np.maximum(0,par.y_low-spouse_inc)
    else:
        y_low_l = par.y_low

    # taxes
    T_c = np.maximum(0,par.tau_c*(taxable_income - y_low_l))
    T_h = np.maximum(0,par.tau_h*(taxable_income - y_low_l))
    T_l = np.maximum(0,par.tau_m*(personal_income - y_low_l))
    T_m = np.maximum(0,par.tau_m*(personal_income - par.y_low_m))
    T_u = np.maximum(0,np.minimum(par.tau_u,par.tau_max)*(personal_income - par.y_low_u))
        
    # return posttax income
    return personal_income - T_c - T_h - T_l - T_m - T_u            

##################################
####       transitions       #####
##################################
@njit(parallel=True)
def oap_pretax(t,par,t_s=-99,y_s=np.nan,d_s=-99): 
    """ old age pension (folkepension) pretax """

    # initialize
    OAP = 0.0

    # check if agent is in oap age
    if par.oap_age <= age(t,par) <= par.end_T:

        # single
        if not par.couple:
            OAP = par.oap_B + par.A_i[0]

        # couple
        else:
                   
            # spousal age
            age_s = age(t_s,par)

            # 1. define index
            if d_s == 1 or age_s < par.oap_age:
                i = 1   # corresponds to 2 in the paper
            elif d_s == 0 and age_s >= par.oap_age:
                i = 2   # correpsonds to 3 in the paper

            # 2. compute OAP_A
            y_h = (i==1)*(y_s - 0.5*np.minimum(par.D_s, y_s))
            OAP_A = np.maximum(0, par.A_i[i] - np.maximum(0, par.tau_i[i]*(y_h - par.D_i[i])))

            # 3. OAP
            OAP = par.oap_B + OAP_A

    # return
    return OAP

@njit(parallel=True)
def erp_pretax(t,ma,st,ra,par):
    """ early retirement pension (efterløn) pretax"""

    # initialize
    ERP = 0.0

    # pre two year period
    if par.T_erp <= t < par.T_two_year:
        if ra == 1:
            priv = priv_pension(ma,st,par)
            ERP = np.maximum(0,par.ERP_high - 0.6*0.05*np.maximum(0, priv - par.ERP_low))

    # two year period
    elif par.T_two_year <= t < par.T_oap:

        # two year rule is satisfied
        if ra == 0:
            ERP = par.ERP_2

        # two year rule not satisfied
        elif ra == 1:
            priv = priv_pension(ma,st,par)
            ERP = np.maximum(0,par.ERP_high - 0.6*0.05*np.maximum(0, priv - par.ERP_low))

    # return 
    return ERP

@njit(parallel=True)
def priv_pension(ma,st,par):
    """ private pension wealth """

    hs = state_translate(st,'high_skilled',par)    
    if ma == 1:
        if hs == 0:
            return par.pension_male[0]
        elif hs == 1:
            return par.pension_male[1]
    elif ma == 0:
        if hs == 0:
            return par.pension_female[0]
        elif hs == 1:
            return par.pension_female[1]


@njit(parallel=True)
def labor_pretax(t,ma,st,par):
    """ pretax labor income """

    # states
    ag = age(t,par)
    hs = state_translate(st,'high_skilled',par)

    # compute
    if ma == 1:
        rlm = par.reg_labor_male    # regression coefficients for men
        log_inc = rlm[0] + rlm[1]*hs + rlm[2]*ag + rlm[3]*(ag**2)/100
    elif ma == 0:
        rlf = par.reg_labor_female  # regression coefficients for women
        log_inc = rlf[0] + rlf[1]*hs + rlf[2]*ag + rlf[3]*(ag**2)/100
    
    # exponentiate and denominate in 100.000 dkr
    return np.exp(log_inc)/par.denom

@njit(parallel=True)
def survival(t,ma,st,par):
    """ survival probability"""    
    
    # states
    ag = age(t,par)
    hs = state_translate(st,'high_skilled',par)

    if ag >= par.end_T:   # dead
        return 0.0
    
    # compute
    else:
        if ma == 1: 
            rsm = par.reg_survival_male
            deadP = np.minimum(1,np.exp(rsm[0] + rsm[1]*ag))
            survivalP = (1-deadP)*((hs==0)*rsm[2] + (hs==1)*rsm[3])
        elif ma == 0:       
            rsf = par.reg_survival_female
            deadP = np.minimum(1,np.exp(rsf[0] + rsf[1]*ag))
            survivalP = (1-deadP)*((hs==0)*rsf[2] + (hs==1)*rsf[3])
        return np.minimum(1,survivalP) 