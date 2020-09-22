import time
import multiprocessing as mp
import pandas as pd
import os
import functools
from KMVmodel import *
from sys import exit


if __name__ == '__main__':

    et = time.time()
    print('Preparing input ....')
    # with pd.HDFStore(os.path.join(OutputFolder,'KMV_Data.h5')) as myData:
    #     df_FIN = myData['df_FIN']
    #     df_STK = myData['df_STK']
    #     ID_KMV = myData['ID_KMV']

    with pd.HDFStore(os.path.join(OutputFolder,'Result.h5')) as myData:
        try:
            myData.keys()
            res = myData['df_Result']
            res = res.set_index(['ID','FINDATE'])
            # 將資料分為金融與非金融
            res['Fin'] = res['TSE 產業別'].apply(lambda x : 1 if x in [17] else 0)
            df_delta = pd.DataFrame(res.groupby(['Fin','FINDATE'])['DELTA'].mean())
        except:
            print('There is no step 1 result data')
            exit(1)

    
    
    Input_pool = [(i[0], 
                   i[1],
                   df_delta.loc[(res.loc[i,'Fin'],i[1])].values[0] ) for i in res.index]

    print('Finished ! It took {} second ....\n'.format(time.time()-et))

    pool = mp.Pool(mp.cpu_count())
    print('Running KMV models ....')
    Result = pool.starmap_async(Main2, Input_pool, callback = functools.partial(handle_output,Name = 'df_Result2')).get()
    pool.close()
    pool.join()