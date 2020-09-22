import numpy as np
from pandas.tseries.offsets import MonthEnd
from scipy import optimize
import time
import multiprocessing as mp
import pandas as pd
import os
import math

InputFolder = 'Input'
OutputFolder = 'Output'

with pd.HDFStore(os.path.join(InputFolder,'econ.h5')) as myData:
    df_R = pd.DataFrame(myData.get('R'))
    df_FX = myData.get('FX')
    
with pd.HDFStore(os.path.join(OutputFolder,'KMV_Data.h5')) as myData:
    df_FIN = myData['df_FIN']
    df_STK = myData['df_STK']
    ID_KMV = myData['ID_KMV']
    
TEND = 1

def Main(ID,  start, end, FIN, STK, TEND =TEND, df_R=df_R):
    import pandas as pd
    start = max(start,pd.to_datetime('2008-01-01'))
    end = min(end,pd.to_datetime('2019-12-31'))

    if start > end :
        print('{} could not compute'.format(ID))
        return ()
    
    et = time.time()
    #FIN = df_FIN.query('統一編號 == "{}"'.format(ID))
    #STK = df_STK.query('統一編號 == "{}"'.format(ID)) 
    TSE_code = FIN['BTYPE'].unique()[0]
    

    FIN = FIN.asfreq('B',method='ffill') 

    joinTable = STK.join(df_R.join(FIN[['TOAS','TOLI','CULI','LTDB']]).fillna(method = 'ffill')).fillna(method = 'ffill')
    
    time_series= [i.strftime('%Y%m%d') for i in pd.date_range(start,end,freq='M')]

    et = time.time()
    print(str(pd.datetime.now())+' PID {} : {} Start To Optimization....'.format(os.getpid(),ID))

    Result = ()
    delta_0 = None

    for i in time_series:
        e = i 
        s = (pd.to_datetime(i) - pd.DateOffset(years = 2))
        
        df_temp = joinTable.query('index > {} and index <= {}'.format(repr(s),repr(e)))

        if len(df_temp) < 250:
            continue
        
        
        Ans = KMVModel(i, df_temp['P'], df_temp['Share'], df_temp[['CULI','LTDB','TOLI','TOAS']], df_temp['數值'], TEND, delta_0)
        Result = tuple([tuple([ID, TSE_code] )  + Ans]) + Result
        delta_0 = Ans[4]
               

    
    print(str(pd.datetime.now())+' PID {} : ID {} has Finished , it cost {} ..... '.format( os.getpid(), ID, time.time() - et))
    
    return Result


def Main2(ID,  date, delta, FIN = df_FIN, STK = df_STK, TEND =TEND, df_R=df_R):
    import pandas as pd

    
    et = time.time()
    FIN = FIN.loc[ID]
    STK = STK.loc[ID]
    TSE_code = FIN['BTYPE'].unique()[0]
    
    FIN = FIN.asfreq('B',method='ffill') 
    joinTable = STK.join(df_R.join(FIN[['TOAS','TOLI','CULI','LTDB']]).fillna(method = 'ffill')).fillna(method = 'ffill')
    

    et = time.time()
    print(str(pd.datetime.now())+' PID {} : {} date {} Start To Optimization....'.format(os.getpid(),ID,date))
    Result = ()

    
    s = (pd.to_datetime(date) - pd.DateOffset(years = 1))

    df_temp = joinTable.query('index > {} and index <= {}'.format(repr(s),repr(date)))

    if len(df_temp) < 50:
        return Result

    
    Ans = KMVModel2(date, df_temp['P'], df_temp['Share'], df_temp[['CULI','LTDB','TOLI','TOAS']], df_temp['數值'], TEND, delta)
    Result = tuple([tuple([ID, TSE_code] )  + Ans]) + Result
  
    print(str(pd.datetime.now())+' PID {} : ID {} date {} has Finished , it cost {} ..... '.format(os.getpid(), ID, date, time.time() - et))
    
    return Result


def KMVModel2(j, S, EQ, L, r, TEND, delta):

    Sig_Acc = 10e-7
    E = EQ * S
    h =  pd.DataFrame(1,index = S.index,dtype=float, columns=['h'])['h']/250
    
    sig = np.log(1 + S.pct_change()).std()*np.sqrt(250)

    
    CL = L['CULI'].values[-1]
    NC = L['LTDB'].values[-1]
    OL = (L['TOLI'] - L['CULI'] - L['LTDB']).values[-1]
    Ln = [CL, NC, OL, L['TOAS']]



    x0 = sig
        
    f = lambda x : -KMVlogLikelihood([x,delta],j,Ln,E,S,r,h,TEND)
    ans = optimize.minimize(f,x0,bounds= [[1e-7,np.inf]])

    sig = ans.x[0]
    
    L_ans = CL + 0.5 * NC + delta * OL

    V_1 = NRMethod(L_ans, L_ans, E[0], r[0], (len(S) - 1)/250 + 1, sig)
    V_n = NRMethod(L_ans, L_ans, E[-1], r[-1], 1, sig)
    mu = mu_bar(sig, h, V_1, V_n, Ln[3][0], Ln[3][-1])
    

    DD = DTD(L_ans, E.iloc[-1], r.iloc[-1], TEND, sig)

    
    return j, ans.success*1, mu, sig, delta, DD



def KMVModel(j, S, EQ, L, r, TEND, delt_0):

    Sig_Acc = 10e-7
    E = EQ * S
    h =  pd.DataFrame(1,index = S.index,dtype=float, columns=['h'])['h']/250

    
    if delt_0 == None:
        delta_bound = [0,1]
        delta = 0
    else:
        delta_bound = [np.max([0,delt_0 - 0.05]),np.min([1,delt_0 + 0.05])]
        delta = delt_0
    
    sig = np.log(1 + S.pct_change()).std()*np.sqrt(250)

    
    CL = L['CULI'].values[-1]
    NC = L['LTDB'].values[-1]
    OL = (L['TOLI'] - L['CULI'] - L['LTDB']).values[-1]
    Ln = [CL, NC, OL, L['TOAS']]



    x0 = [sig,delta]
        
    f = lambda x : -KMVlogLikelihood(x,j,Ln,E,S,r,h,TEND)
    ans = optimize.minimize(f,x0,bounds= [[1e-7,np.inf],delta_bound])

    sig = ans.x[0]
    delta = ans.x[1]
    L_ans = CL + 0.5 * NC + delta * OL

    V_1 = NRMethod(L_ans, L_ans, E[0], r[0], (len(S) - 1)/250 + 1, sig)
    V_n = NRMethod(L_ans, L_ans, E[-1], r[-1], 1, sig)
    mu = mu_bar(sig, h, V_1, V_n, Ln[3][0], Ln[3][-1])
    

    DD = DTD(L_ans, E.iloc[-1], r.iloc[-1], TEND, sig)

    
    return j, ans.success*1, mu, sig, delta, DD


def d_t(V, L, sig, r, T):

    return (np.log(V/L) + (r + (sig**2)/ 2) * T) / (sig * np.sqrt(T))

def mu_bar(sig, h, V_1, V_n, A_1, A_n):

    return sig ** 2 * 0.5 + 1/(sum(h[1:])) * np.log((V_n * A_1) / (A_n * V_1))


def KMVlogLikelihood(X, j, L, E, S ,r, h, TEND):
    
    sig = X[0]
    delta = X[1]

    n = len(S)
    
    TV =  [(len(S.loc[i:j])-1)/250 for i in S.index]

    CL, NC, OL, A = L[0], L[1], L[2], L[3]
    L = CL + 0.5 * NC + delta * OL
    V = [ NRMethod(L, L ,e , ri , tv + TEND, sig) for e, tv,  ri in zip(E,TV,r)]

    mu = mu_bar(sig, h, V[0], V[-1], A[0], A[-1])
    
    ML1 = (n-1)/2 * np.log(2 * np.pi)
    ML2 = sum( np.log(h[1:].values  * sig ** 2 )) * 0.5
    ML3 = sum(np.log(V/A)[1:])
    ML4 = sum([np.log( 0.5 * (1 + math.erf(d_t(v, L, sig, ri, tv + TEND)/np.sqrt(2)))) for v,  tv, ri in zip(V,  TV, r)][1:])
    
    A_p = A[:-1].values / A[1:].values
    V_p = np.array(V[1:]) / np.array(V[:-1])

    ML5 = sum([ ((np.log(vi * ai)-(mu-(sig ** 2 )/2) * hi)**2) / (2 * hi * sig ** 2) for hi, vi, ai in zip(h[1:], V_p, A_p)])

    return -(ML1 + ML2 + ML3 + ML4 + ML5)

def DTD(L, E, r, T, sig):
    
    V = NRMethod(L, L ,E , r , T, sig)
    return np.log(V/L) / (sig * np.sqrt(T))
    

def NRMethod(V,L,E,r,T,sig):
    
    
    N = 1000000
    NR_Acc=10e-7
    Tem=0
    k=1
    while abs(V-Tem)>NR_Acc and k < N:
        
        d1=(np.log(V/L)+(r+(0.5*sig**2))*T)/(sig*np.sqrt(T))
        d2=d1-(sig*np.sqrt(T))
        f=E-V*0.5*(1+math.erf(d1/np.sqrt(2)))+np.exp(-r*(T))*L*0.5*(1+math.erf(d2/np.sqrt(2)))
        df=-0.5*(1+math.erf(d1/np.sqrt(2)))   # the derivative
        Tem=V
        V=Tem-f/df
        k=k+1

    return V


def handle_output(Result,Name):
    myData = pd.HDFStore(os.path.join('Output','Result.h5'),mode='a')
    for r in Result:
        Result = pd.DataFrame(list(r),columns = ['ID','TSE 產業別','FINDATE','Success','MU','SIG','DELTA','DD'])
        myData.append(Name,Result,format='table')
    myData.close()








