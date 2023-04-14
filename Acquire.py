import pandas as pd
import os 
def acquire_data():
    
    if os.path.exists('wine_data.csv'):
        df = pd.read_csv('wine_data.csv')
    else:
        df_1 = pd.read_csv('https://query.data.world/s/s7hsvetdhmtlkdq4x6ofz3wd5snwut?dws=00000')
        df_2 = pd.read_csv('https://query.data.world/s/dth4xlpnfu3lnyln2lxfdms4vxf3hl?dws=00000')
        df_1['type_of_wine'] = ['Red'] * len(df_1)
        df_2['type_of_wine'] = ['White'] * len(df_2)
        combine = [df_1,df_2]
        df = pd.concat(combine)
        df.to_csv('wine_data.csv',index=False)
    
    return df