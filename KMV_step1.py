import time
import multiprocessing as mp
import pandas as pd
import os
import functools
from KMVmodel import *


    
if __name__ == '__main__':


    NN = len(ID_KMV)
    n = 100
    tt = NN//n

    for j in range(2,tt+1):
        
        s = n * j
        e = min(n*j+n,NN)

        et = time.time()
        print('Preparing input ....')
        Input_pool = [ ( i,
                        ID_KMV.loc[i,'KMV_start'], 
                        ID_KMV.loc[i,'KMV_end'],
                        df_FIN.loc[i],
                        df_STK.loc[i],
                        ) for i in ID_KMV.index]

        print('Finished ! It took {} second ....\n'.format(time.time()-et))

        pool = mp.Pool(mp.cpu_count())
        print('Running KMV models ....')
        Result = pool.starmap_async(Main, Input_pool, callback = functools.partial(handle_output,Name = 'df_Result')).get()
        pool.close()
        pool.join()


