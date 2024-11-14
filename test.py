import numpy as np


def discrete_power_law(n, alpha, beta):
    half_n = int(n/2)
    C_beta_half_n = 0
    for i in range(1, half_n+1):
        C_beta_half_n += i**(-beta)
    probability_alpha = C_beta_half_n**(-1) * alpha**(-beta)
    return probability_alpha


weights = []
for mutation_rate in [2, 5, 10, 20, 40]:
    weights.append(discrete_power_law(5, mutation_rate, 1.5))
weights = np.array(weights)
weights /= weights.sum()
print(weights)
print(weights.sum())
