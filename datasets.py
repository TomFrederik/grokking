from itertools import permutations, product
from math import factorial

import numpy as np
import torch


class S5Data(torch.utils.data.Dataset):
    def __init__(self, data_path=None):
        assert data_path is not None, "data_path is None"
        self.data = np.load(data_path)
        print(self.data.shape)
        
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

if __name__ == "__main__":
    S5Data.generate_data('./data/s5.npy')
    # dataset = S5Data('./data/s5.npy')