import matplotlib.pyplot as plt
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import pickle
import os, cv2
from preprocessing import parse_annotation, BatchGenerator
from utils import WeightReader, decode_netout, draw_boxes
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import random 
import gc

LABELS =  ['species_fourspot',
           'species_grey sole',
           'species_other', 
           'species_plaice',
           'species_summer',
           'species_windowpane',
           'species_winter']

IMAGE_H, IMAGE_W = 416, 416
GRID_H,  GRID_W  = 13 , 13
BOX              = 5
CLASS            = len(LABELS)
CLASS_WEIGHTS    = np.ones(CLASS, dtype='float32')
THRESHOLD        = 0.3
ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

NO_OBJECT_SCALE  = 1.0
OBJECT_SCALE     = 5.0
COORD_SCALE      = 1.0
CLASS_SCALE      = 1.0

BATCH_SIZE       = 12 * 2
WARM_UP_BATCHES  = 100
# TRUE_BOX_BUFFER  = 50
TRUE_BOX_BUFFER  = 2
REMOVE_NEGATIVE_ITEMS = 0.995
AUG_FREQ = 0.2
VALID_SHARE = 0.2
LAYER_SCRATCH = ['conv_20','conv_21','conv_22','conv_23']


generator_config = {
    'IMAGE_H'         : IMAGE_H, 
    'IMAGE_W'         : IMAGE_W,
    'GRID_H'          : GRID_H,  
    'GRID_W'          : GRID_W,
    'BOX'             : BOX,
    'LABELS'          : LABELS,
    'CLASS'           : len(LABELS),
    'ANCHORS'         : ANCHORS,
    'BATCH_SIZE'      : BATCH_SIZE,
    'TRUE_BOX_BUFFER' : 2,
}


def change_coords(x1,y1,x2,y2,f_len):
    
    max_x = 1280
    max_y = 720
    
    x_av = (x2+x1)/2
    y_av = (y2+y1)/2
    

    if(x_av-f_len/2)>max_x:
        x1_new = max_x
    elif (x_av-f_len/2)<0:
        x1_new = 0
    else:
        x1_new=x_av-f_len/2 
        
    if(x_av+f_len/2)>max_x:
        x2_new = max_x
    else:
        x2_new=x_av+f_len/2 
        
    if(y_av-f_len/2)>max_y:
        y1_new = max_y
    elif (y_av-f_len/2)<0:
        y1_new = 0
    else:
        y1_new=y_av-f_len/2 

    if(y_av+f_len/2)>max_y:
        y2_new = max_y
    else:
        y2_new=y_av+f_len/2

    return x1_new,y1_new,x2_new,y2_new

df = pd.read_csv('train.csv')

df['len_bucket_100'] = pd.cut(df.length, 100, labels=False)
df['len_bucket_50'] = pd.cut(df.length, 50, labels=False)
df['len_bucket_200'] = pd.cut(df.length, 200, labels=False)
df['len_bucket_20'] = pd.cut(df.length, 20, labels=False)

df['_'] = df.apply(lambda row: change_coords(row['x1'],row['y1'],row['x2'],row['y2'],row['length']), axis=1)
df[['x1_new','y1_new','x2_new','y2_new']] = df['_'].apply(pd.Series)
del df['_']

df['x_d'] = df['x2_new'] - df['x1_new']
df['y_d'] = df['y2_new'] - df['y1_new']

from sklearn.linear_model import LinearRegression

X = df[df['x_d']>0][['x_d','y_d']].values
y = df[df['x_d']>0]['length'].values

clr = LinearRegression()
clr.fit(X,y)

del df,X,y

from tqdm import tqdm
import pandas as pd
import numpy as np
from tqdm import tqdm

df_sub = pd.read_csv('submission_vec_ths_10.csv')
del df_sub['Unnamed: 0']

threshold = 0.2
block_high_prob_threshold = 0.75

LABELS =  ['species_fourspot',
           'species_grey sole',
           'species_other', 
           'species_plaice',
           'species_summer',
           'species_windowpane',
           'species_winter']

box_list = ['xmin','xmax', 'ymin', 'ymax']

def filter_dataframe(df,threshold):
    df.loc[df_sub[LABELS].sum(axis=1)<threshold,LABELS+box_list] = 0    
    return df

def get_block(df,idx_start,idx_end):
    return df[LABELS+box_list].loc[idx_start:idx_end-1]

def process_block(block):
    block_start_offset = np.argmax(block[LABELS].sum(axis=1).values)
    block_max_value = np.max(block[LABELS].sum(axis=1).values)
    block_list = list(block.loc[block.index.values.min()+block_start_offset:,LABELS].sum(axis=1)>block_high_prob_threshold*block_max_value)
    try: 
        block_length = block_list.index(False) 
    except: 
        block_length = block.shape[0] - block_start_offset
    
    return block_start_offset,block_max_value,block_length

def update_block(df,
                 idx_start,idx_end,
                 block_start_offset,block_length,
                 fish_number):
    
    # print (df.loc[idx_start:idx_end][LABELS])
    # do some tweaking here
    if block_start_offset>0:
        df.loc[idx_start:idx_start+block_start_offset-1,LABELS+box_list] = 0
    # print (df.loc[idx_start:idx_end][LABELS])
    
    df.loc[idx_start+block_start_offset+block_length:idx_end,LABELS+box_list] = 0
    # print (df.loc[idx_start:idx_end][LABELS])    
    df.loc[idx_start+block_start_offset:idx_start+block_start_offset+block_length-1,'fish_number'] = fish_number
    # print (df.loc[idx_start:idx_end][LABELS])    
    
    
    
df_sub = filter_dataframe(df_sub,threshold)

# df_sub = df_sub[df_sub.video_id.isin(['01rFQwp0fqXLHg33','09WWcMSr5nbKk0lb'])]
# df_sub = df_sub.loc[18392:21261]

current_idx =  0
block_start_idx = -1
block_end_idx = 0
is_block = 0
fish_number = 0
previous_vid = 'dfsdfsdfsdfsdfsddfs'
current_vid = 'asfsadfsdafdsafasdfsdfa'


with tqdm(total = df_sub.shape[0]) as pbar:
    for index, row in df_sub.iterrows():
        
        current_vid = row['video_id']
        
        # print(current_idx)
        if(row['xmin']>0): 
            is_block = 1
            # print('is block triggered')
            # print(current_vid,current_idx,is_block,block_start_idx,block_end_idx,block_just_started)
        else:
            is_block = 0

        if ((is_block == 1) & (block_start_idx == -1)):
            block_start_idx = current_idx
            block_just_started = 1
            # print('block_start_idx triggered')
            # print(current_vid,current_idx,is_block,block_start_idx,block_end_idx,block_just_started)           

        if((block_start_idx > -1) & (is_block==0) & (df_sub.loc[current_idx:current_idx+5,'xmin'].sum()==0) & (current_vid == previous_vid)):
            block_end_idx = current_idx
            # print('block_end_idx triggered')
            # print(current_vid,current_idx,is_block,block_start_idx,block_end_idx,block_just_started)                      
        
        if((block_start_idx < current_idx-1) & (is_block==1) & (current_vid != previous_vid)):
            block_end_idx = current_idx 
            # print('block_end_idx triggered')
            # print(current_vid,current_idx,is_block,block_start_idx,block_end_idx,block_just_started)
        
        if (block_end_idx>0):
            # print(current_vid,current_idx,is_block,block_start_idx,block_end_idx,block_just_started)      
            block = get_block(df_sub,block_start_idx,block_end_idx)
            # print(block_start_idx,block_end_idx)
            block_start_offset,block_max_value,block_length =  process_block(block)
            del block
            fish_number += 1
            update_block(df_sub,block_start_idx,block_end_idx,block_start_offset,block_length,fish_number)
            gc.collect()
            
            if current_vid != previous_vid:
                fish_number = 0
                
            if current_vid != previous_vid:
                block_start_idx = current_idx
            else:
                block_start_idx = -1
            
            block_end_idx = 0
            # print('block update triggered')        
        current_idx += 1
        previous_vid = row['video_id']
        # print(current_vid,previous_vid,fish_number)

        pbar.update(1) 
 
df_sub['x_d'] = df_sub['xmax'] - df_sub['xmin']
df_sub['y_d'] = df_sub['ymax'] - df_sub['ymin']
preds = clr.predict(df_sub[df_sub['x_d']>0][['x_d','y_d']])
df_sub.loc[df_sub['x_d']>0,'length'] = preds

df_sub.to_csv('submissions_model_vector_2.csv')

del df_sub['xmin'],df_sub['xmax'],df_sub['ymin'],df_sub['ymax'],df_sub['x_d'],df_sub['y_d']
df_sub = df_sub.set_index('row_id')

df_sub.to_csv('submissions_model_vector_3.csv')