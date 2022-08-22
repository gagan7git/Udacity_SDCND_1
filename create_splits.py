import argparse
import glob
import os
import random

import numpy as np

from utils import get_module_logger
import shutil


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /home/workspace/data
    """
    
    # TODO: Split the data present in `/home/workspace/data/waymo/training_and_validation` into train and val sets.
    try:
        files = [filename for filename in glob.glob(f'{data_dir}/waymo/training_and_validation/*.tfrecord')]
    except:
        print("Issue in TF record file parse")
    np.random.shuffle(files)	
    train_files, val_file = np.split(files, [int(.80*len(files))])
    # You should move the files rather than copy because of space limitations in the workspace.
	
    train = os.path.join(data_dir, 'train')
    val = os.path.join(data_dir, 'val') 

    try:	
        if os.path.exists(train):
            os.makedirs(train)
        os.makedirs(train,exist_ok=True)   
        
        if os.path.exists(val):
            os.makedirs(val)
        os.makedirs(val,exist_ok=True) 
    except:
        print("pass")	

	
    for file in train_files:
        shutil.move(file, train)   
    for file in val_file:
        shutil.move(file, val)	

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)