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
    return {
        'plus': XpYData(data_dir, force_data, num_elements),
        'minus': XminYData(data_dir, force_data, num_elements),
        'perm': SNData(data_dir, force_data, num_elements)
    }[descr]

class XpYData(torch.utils.data.Dataset):
    def __init__(self, data_dir=None, force_data=False, prime=97):
        assert data_dir is not None, "data_dir is None"
        assert isPrime(prime), "prime is not prime"
        if force_data:
            logging.info(f"Creating data and saving to {data_dir}")
            self.generate_data(data_dir, prime)
        logging.info(f"Loading data from {data_dir}")
        self.data = np.load(os.path.join(data_dir, f'xpy{prime}.npy'))
    
    def __getitem__(self, index):
        return np.array(self.data[index])

    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def generate_data(data_dir, prime=97):
        data = []
        op = prime
        eq = prime + 1
        
        all_permutations = list(permutations(range(prime)))
        for i,j in product(range(prime), repeat=2):
            res = (i + j) % prime
            data.append([i, op, j, eq, res])
        
        # save data
        np.save(os.path.join(data_dir, f'xpy{prime}.npy'), data)

class XminYData(torch.utils.data.Dataset):
    def __init__(self, data_dir=None, force_data=False, prime=97):
        assert data_dir is not None, "data_dir is None"
        assert isPrime(prime), "prime is not prime"
        if force_data:
            logging.info(f"Creating data and saving to {data_dir}")
            self.generate_data(data_dir, prime)
        logging.info(f"Loading data from {data_dir}")
        self.data = np.load(os.path.join(data_dir, f'xminy{prime}.npy'))
    
    def __getitem__(self, index):
        return np.array(self.data[index])

    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def generate_data(data_dir, prime=97):
        data = []
        op = prime
        eq = prime + 1
        
        all_permutations = list(permutations(range(prime)))
        for i,j in product(range(prime), repeat=2):
            res = (i - j) % prime
            data.append([i, op, j, eq, res])
        
        # save data
        np.save(os.path.join(data_dir, f'xminy{prime}.npy'), data)


class SNData(torch.utils.data.Dataset):
    def __init__(self, data_dir=None, force_data=False, group_size=5):
        assert data_dir is not None, "data_dir is None"
        if force_data:
            logging.info(f"Creating data and saving to {data_dir}")
            self.generate_data(data_dir, group_size)
        logging.info(f"Loading data from {data_dir}")
        self.data = np.load(os.path.join(data_dir, f's{group_size}.npy'))
        
    def __getitem__(self, index):
        return np.array(self.data[index])
        
    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def generate_data(data_dir, group_size=5):
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
        np.save(os.path.join(data_dir, f's{group_size}.npy'), data)