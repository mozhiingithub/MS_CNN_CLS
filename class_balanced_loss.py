'''
Class-Balanced Loss Based on Effective Number of Samples
'''
import numpy as np


def get_beta(unique_prototypes: int):
    if 0 == unique_prototypes:
        return 0.9999
    else:
        return float((unique_prototypes - 1) / unique_prototypes)


def get_effective_number(beta, n):
    return (1.0 - beta ** n) / (1.0 - beta)


def get_factor(effective_number):
    return 1.0 / effective_number


def get_factor_list(sample_nums, unique_prototypes=None):
    class_num = len(sample_nums)
    if unique_prototypes is None:
        unique_prototypes = [0 for i in range(class_num)]
    betas = [get_beta(up) for up in unique_prototypes]
    e_n_ls = [get_effective_number(betas[i], sample_nums[i]) for i in range(class_num)]
    factors = [get_factor(e_n_ls[i]) for i in range(class_num)]
    factors = np.array(factors, dtype=np.float32)
    return factors


# init_nums = np.array([293, 51], dtype=np.int)
# trn_init_nums = (0.8 * init_nums).astype(np.int)
# aug_nums = 80 * trn_init_nums
#
# print(init_nums)
# print(trn_init_nums)
# print(aug_nums)
# print()
# print(get_factor_list(sample_nums=aug_nums))
# print(get_factor_list(sample_nums=aug_nums, unique_prototypes=trn_init_nums))
# print(1.0 / trn_init_nums)
# print(1.0 / aug_nums)
