from model.pins_model import train_net
from pathlib import Path
import pickle
from numpy.random import default_rng
import numpy as np

def to_binary_class(lab):
    d={'Epithelioid': 0, 'Biphasic': 1, 'sarcomatoid': 1}
    return d[lab]

def stratified_split(df, fracs=[0.6, 0.2, 0.2], shuffle=False, seed=None):
    #fracs defines split ratios
    split_fracs=np.cumsum(fracs)
    rng = default_rng(seed)
    label_inds=[df[df.labels==l].index for l in df.labels.unique()]
    df_tr, df_val, df_test=[],[],[]
    sarc_in_tr=False   #if want to force sarc examples into train
    for i, inds in enumerate(label_inds):
        if i==2 and sarc_in_tr:
            print('putting sarc examples into train')
            #force our 2 sarc examples to be in train
            df_tr.extend(inds)
            continue
        split_pts=(np.array(split_fracs)*len(inds)).astype(int)
        if shuffle:
            temp=np.array(inds)
            rng.shuffle(temp)
            temp=np.array_split(temp, split_pts)
        else:
            temp=np.array_split(inds, split_pts)
        df_tr.extend(temp[0])
        df_val.extend(temp[1])
        df_test.extend(temp[2])
    df_tr=df.loc[df_tr]
    df_val=df.loc[df_val]
    df_test=df.loc[df_test]
    dfs={'Train': df_tr, 'Val': df_val, 'Test': df_test}
    return dfs

if __name__=='__main__':
    base_path=Path(r'D:\TCGA_Data\Local_experiments\TMA_MIL_nomil')
    base_path.mkdir()
    
    with open(Path('E:\TCGA_Data\MESOv_hdf5\TMA_labels.pkl'), 'rb') as loadfile:
        unpickler = pickle.Unpickler(loadfile)
        df=unpickler.load()
        loadfile.close()

    for i in range(4):
        p=base_path.joinpath(f'Temp{i}')
        p.mkdir()
        tr_val=df[df['slide']!=i+1]
        dfs=stratified_split(tr_val, fracs=[0.75,0.25,0], shuffle=True)
        dfs['Test']=df[df['slide']==i]
        p=base_path.joinpath(f'Temp{i}')
        train_net(dfs,p)










