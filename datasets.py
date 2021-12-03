from itertools import permutations, product
import logging
from math import factorial
import os

import numpy as np
import torch

def isPrime(n):
    if n & 1 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d = d + 2
    return True

def get_dataset(descr, num_elements, data_dir=None, force_data=False):
    if not descr.startswith('perm'):
        return ArithmeticData(data_dir, force_data, num_elements, descr)
    else:
        return PermData(data_dir, force_data, num_elements, descr)

def get_arithmetic_func(func_name):
    return {
        'plus': lambda x,y,p: (x + y) % p,
        'minus': lambda x,y,p: (x - y) % p,
        # div #TODO
        # div_odd #TODO
        'x2y2': lambda x,y,p: (x ** 2 + y ** 2) % p,
        'x2xyy2': lambda x,y,p: (x ** 2 + x * y + y ** 2) % p,
        'x2xyy2x': lambda x,y,p: (x ** 2 + x * y + y ** 2 + x) % p,
        'x3xy': lambda x,y,p: (x ** 3 + x * y) % p,
        'x3xy2y': lambda x,y,p: (x ** 3 + x * y ** 2 + y) % p
    }[func_name]

class ArithmeticData(torch.utils.data.Dataset):
    def __init__(self, data_dir=None, force_data=False, prime=97, func_name="plus"):
        assert data_dir is not None, "data_dir is None"
        assert isPrime(prime), "prime is not prime"
        if force_data:
            logging.info(f"Creating data and saving to {data_dir}")
            self.generate_data(data_dir, func_name, prime)
        logging.info(f"Loading data from {data_dir}")
        self.data = np.load(os.path.join(data_dir, f'{func_name}_{prime}.npy'))
    
    def __getitem__(self, index):
        return np.array(self.data[index])

    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def generate_data(data_dir, func_name, prime=97):
        data = []
        op = prime
        eq = prime + 1
        
        all_permutations = list(permutations(range(prime)))
        for x, y in product(range(prime), repeat=2):
            res = get_arithmetic_func(func_name)(x, y, prime)
            data.append([x, op, y, eq, res])
        
        # save data
        np.save(os.path.join(data_dir, f'{func_name}_{prime}.npy'), data)


class PermData(torch.utils.data.Dataset):
    def __init__(self, data_dir=None, force_data=False, group_size=5, func_name="perm_xy"):
        assert data_dir is not None, "data_dir is None"
        if force_data:
            logging.info(f"Creating data and saving to {data_dir}")
            self.generate_data(data_dir, group_size, func_name)
        logging.info(f"Loading data from {data_dir}")
        self.data = np.load(os.path.join(data_dir, f'{func_name}_{group_size}.npy'))
        
    def __getitem__(self, index):
        return np.array(self.data[index])
        
    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def generate_data(data_dir, group_size=5, func_name="perm_xy"):
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
        np.save(os.path.join(data_dir, f'{func_name}_{group_size}.npy'), data)