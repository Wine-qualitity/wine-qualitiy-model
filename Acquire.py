import pandas as pd

def acquire_data():
    df_1 = pd.read_csv('https://query.data.world/s/s7hsvetdhmtlkdq4x6ofz3wd5snwut?dws=00000')
    df_2 = pd.read_csv('https://query.data.world/s/dth4xlpnfu3lnyln2lxfdms4vxf3hl?dws=00000')
    df_1['type_of_wine'] = ['Red'] * len(df_1)
    df_2['Type_of_wine'] = ['White'] * len(df_2)
    combine = [df_1,df_2]
    df = pd.concat(combine)
    df