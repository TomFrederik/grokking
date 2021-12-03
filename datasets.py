from itertools import permutations, product
import logging
from math import factorial

import numpy as np
import torch


class S5Data(torch.utils.data.Dataset):
    def __init__(self, data_path=None, force_data=False):
        assert data_path is not None, "data_path is None"
        if force_data:
            logging.info(f"Creating data and saving to {data_path}")
        logging.info(f"Loading data from {data_path}")
        self.data = np.load(data_path)
        
    def __getitem__(self, index):
        return np.array(self.data[index])
        
    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def generate_data(data_path, group_size=5):
        data = []
        op = group_size
        eq = group_size + 1
        
        all_permutations = list(permutations(range(group_size)))
        for i,j in product(range(factorial(group_size)), repeat=2):
            # compute result of combining i and j
            i_, j_ = all_permutations[i], all_permutations[j]
            combined = []
            for k in range(len(i_)):
                combined.append(j_.index(i_[k]))
            combined = tuple(combined)
            res = all_permutations.index(combined)
            data.append([i, op, j, eq, res])
        
        # save data
        np.save(data_path, data)