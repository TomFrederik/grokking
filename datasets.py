from itertools import permutations, product
import logging
from math import factorial
import os

import numpy as np
import torch

###
# some utility functions
###

def isPrime(n):
    """
    Checks whether n is a prime number
    """
    if n & 1 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d = d + 2
    return True

def get_inverse_perm(perm):
    """
    Computes inverse of a given permutation.
    """
    perm = np.array(perm)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm), dtype=perm.dtype)
    return list(inv)

def compose_perms(perm1, perm2):
    """
    Computes perm1(perm2)
    """
    perm1 = np.array(perm1)
    perm2 = np.array(perm2)
    return tuple(perm1[perm2])

def get_dataset(descr, num_elements, data_dir=None, force_data=False):
    if not descr.startswith('perm'):
        return ArithmeticData(data_dir, force_data, num_elements, descr)
    else:
        return PermData(data_dir, force_data, num_elements, descr)

def get_arithmetic_func(func_name):
    return {
        'plus': lambda x,y,p: (x, y, (x + y) % p),
        'minus': lambda x,y,p: (x, y, (x - y) % p),
        'div': lambda x,y,p: ((x * y) % p, y, x),
        'div_odd': lambda x,y,p: (x, y, (x // y) % p if y % 2 == 1 else (x - y) % p),
        'x2y2': lambda x,y,p: (x, y, (x ** 2 + y ** 2) % p),
        'x2xyy2': lambda x,y,p: (x, y, (x ** 2 + x * y + y ** 2) % p),
        'x2xyy2x': lambda x,y,p: (x, y, (x ** 2 + x * y + y ** 2 + x) % p),
        'x3xy': lambda x,y,p: (x, y, (x ** 3 + x * y) % p),
        'x3xy2y': lambda x,y,p: (x, y, (x ** 3 + x * y ** 2 + y) % p)
    }[func_name]


###
# Dataset classes
###

class ArithmeticData(torch.utils.data.Dataset):
    def __init__(self, data_dir=None, force_data=False, prime=97, func_name="plus"):
        assert data_dir is not None, "data_dir is None"
        assert isPrime(prime), "prime is not prime"

        if force_data:
            logging.info(f"Creating data and saving to {data_dir}")
            self.generate_data(data_dir, func_name, prime)
        logging.info(f"Loading data from {data_dir}")

        try:
            self.data = np.load(os.path.join(data_dir, f'{func_name}_{prime}.npy'))
        except FileNotFoundError:
            path = os.path.join(data_dir, f'{func_name}_{prime}.npy')
            raise FileNotFoundError(f"Could not find {path}. Run with force_data=True to generate data")
    
    def __getitem__(self, index):
        return np.array(self.data[index])

    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def generate_data(data_dir, func_name, prime=97):
        data = []
        func = get_arithmetic_func(func_name)
        op = prime
        eq = prime + 1
        
        if func_name == 'div': # avoid dividing by zero
            y_range = range(1, prime)
        else:
            y_range = range(prime)
        for x, y in product(range(prime), y_range):
            x, y, res = func(x, y, prime)
            data.append([x, op, y, eq, res])
        
        # save data
        np.save(os.path.join(data_dir, f'{func_name}_{prime}.npy'), data)


class PermData(torch.utils.data.Dataset):
    def __init__(self, data_dir=None, force_data=False, group_size=5, func_name="perm_xy"):
        assert data_dir is not None, "data_dir is None"
        assert group_size <= 10, "group_size should not be > 10, otherwise you will run out of RAM"

        if force_data:
            logging.info(f"Creating data and saving to {data_dir}")
            self.generate_data(data_dir, group_size, func_name)
        logging.info(f"Loading data from {data_dir}")

        try:
            self.data = np.load(os.path.join(data_dir, f'{func_name}_{group_size}.npy'))
        except FileNotFoundError:
            path = os.path.join(data_dir, f'{func_name}_{group_size}.npy')
            raise FileNotFoundError(f"Could not find {path}. Run with force_data=True to generate data")
        
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
        for i, j in product(range(factorial(group_size)), repeat=2):
            
            # compose permutations
            perm1, perm2 = all_permutations[i], all_permutations[j]
            combined = compose_perms(perm1, perm2)
            if func_name == "perm_xyx":
                combined = compose_perms(combined, perm1)
            elif func_name == "perm_xyx1":
                combined = compose_perms(combined, get_inverse_perm(perm1))
            
            # get resulting index and save data
            res = all_permutations.index(combined)
            data.append([i, op, j, eq, res])
        
        # save data
        np.save(os.path.join(data_dir, f'{func_name}_{group_size}.npy'), data)
        


