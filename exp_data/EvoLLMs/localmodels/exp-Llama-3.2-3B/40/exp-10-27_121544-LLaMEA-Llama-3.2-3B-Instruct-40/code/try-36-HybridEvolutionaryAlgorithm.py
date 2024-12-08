import numpy as np
from scipy.optimize import differential_evolution
import random

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        if func.__name__ == 'f1':
            return differential_evolution(func, self.bounds)
        elif func.__name__ == 'f2':
            return differential_evolution(func, self.bounds, x0=self.x0)
        elif func.__name__ == 'f3':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5)
        elif func.__name__ == 'f4':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0)
        elif func.__name__ == 'f5':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx')
        elif func.__name__ == 'f6':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5)
        elif func.__name__ == 'f7':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5)
        elif func.__name__ == 'f8':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5)
        elif func.__name__ == 'f9':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1')
        elif func.__name__ == 'f10':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform')
        elif func.__name__ == 'f11':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='log-uniform')
        elif func.__name__ == 'f12':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='log-uniform', scaling='uniform')
        elif func.__name__ == 'f13':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform')
        elif func.__name__ == 'f14':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform')
        elif func.__name__ == 'f15':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f16':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f17':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f18':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f19':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f20':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f21':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f22':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f23':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f24':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform')

    def hybridize(self, func):
        new_individual = func(self.x0)
        if random.random() < 0.4:
            new_individual[0] += random.uniform(-1, 1)
        if random.random() < 0.4:
            new_individual[1] += random.uniform(-1, 1)
        return new_individual

    def optimize(self, func):
        for _ in range(self.budget):
            new_individual = self.hybridize(func)
            func(new_individual)

# Usage
def f1(x):
    return (x[0] - 2)**2 + (x[1] - 2)**2

def f2(x):
    return (x[0] - 1)**2 + (x[1] - 3)**2

def f3(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2

def f4(x):
    return (x[0] - 2)**2 + (x[1] - 3)**2

def f5(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2 + (x[2] - 3)**2

def f6(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2 + (x[2] - 3)**2 + (x[3] - 4)**2

def f7(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2 + (x[2] - 3)**2 + (x[3] - 4)**2 + (x[4] - 5)**2

def f8(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2 + (x[2] - 3)**2 + (x[3] - 4)**2 + (x[4] - 5)**2 + (x[5] - 6)**2

def f9(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2 + (x[2] - 3)**2 + (x[3] - 4)**2 + (x[4] - 5)**2 + (x[5] - 6)**2 + (x[6] - 7)**2

def f10(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2 + (x[2] - 3)**2 + (x[3] - 4)**2 + (x[4] - 5)**2 + (x[5] - 6)**2 + (x[6] - 7)**2 + (x[7] - 8)**2

def f11(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2 + (x[2] - 3)**2 + (x[3] - 4)**2 + (x[4] - 5)**2 + (x[5] - 6)**2 + (x[6] - 7)**2 + (x[7] - 8)**2 + (x[8] - 9)**2

def f12(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2 + (x[2] - 3)**2 + (x[3] - 4)**2 + (x[4] - 5)**2 + (x[5] - 6)**2 + (x[6] - 7)**2 + (x[7] - 8)**2 + (x[8] - 9)**2 + (x[9] - 10)**2

def f13(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2 + (x[2] - 3)**2 + (x[3] - 4)**2 + (x[4] - 5)**2 + (x[5] - 6)**2 + (x[6] - 7)**2 + (x[7] - 8)**2 + (x[8] - 9)**2 + (x[9] - 10)**2 + (x[10] - 11)**2

def f14(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2 + (x[2] - 3)**2 + (x[3] - 4)**2 + (x[4] - 5)**2 + (x[5] - 6)**2 + (x[6] - 7)**2 + (x[7] - 8)**2 + (x[8] - 9)**2 + (x[9] - 10)**2 + (x[10] - 11)**2 + (x[11] - 12)**2

def f15(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2 + (x[2] - 3)**2 + (x[3] - 4)**2 + (x[4] - 5)**2 + (x[5] - 6)**2 + (x[6] - 7)**2 + (x[7] - 8)**2 + (x[8] - 9)**2 + (x[9] - 10)**2 + (x[10] - 11)**2 + (x[11] - 12)**2 + (x[12] - 13)**2

def f16(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2 + (x[2] - 3)**2 + (x[3] - 4)**2 + (x[4] - 5)**2 + (x[5] - 6)**2 + (x[6] - 7)**2 + (x[7] - 8)**2 + (x[8] - 9)**2 + (x[9] - 10)**2 + (x[10] - 11)**2 + (x[11] - 12)**2 + (x[12] - 13)**2 + (x[13] - 14)**2

def f17(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2 + (x[2] - 3)**2 + (x[3] - 4)**2 + (x[4] - 5)**2 + (x[5] - 6)**2 + (x[6] - 7)**2 + (x[7] - 8)**2 + (x[8] - 9)**2 + (x[9] - 10)**2 + (x[10] - 11)**2 + (x[11] - 12)**2 + (x[12] - 13)**2 + (x[13] - 14)**2 + (x[14] - 15)**2

def f18(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2 + (x[2] - 3)**2 + (x[3] - 4)**2 + (x[4] - 5)**2 + (x[5] - 6)**2 + (x[6] - 7)**2 + (x[7] - 8)**2 + (x[8] - 9)**2 + (x[9] - 10)**2 + (x[10] - 11)**2 + (x[11] - 12)**2 + (x[12] - 13)**2 + (x[13] - 14)**2 + (x[14] - 15)**2 + (x[15] - 16)**2

def f19(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2 + (x[2] - 3)**2 + (x[3] - 4)**2 + (x[4] - 5)**2 + (x[5] - 6)**2 + (x[6] - 7)**2 + (x[7] - 8)**2 + (x[8] - 9)**2 + (x[9] - 10)**2 + (x[10] - 11)**2 + (x[11] - 12)**2 + (x[12] - 13)**2 + (x[13] - 14)**2 + (x[14] - 15)**2 + (x[15] - 16)**2 + (x[16] - 17)**2

def f20(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2 + (x[2] - 3)**2 + (x[3] - 4)**2 + (x[4] - 5)**2 + (x[5] - 6)**2 + (x[6] - 7)**2 + (x[7] - 8)**2 + (x[8] - 9)**2 + (x[9] - 10)**2 + (x[10] - 11)**2 + (x[11] - 12)**2 + (x[12] - 13)**2 + (x[13] - 14)**2 + (x[14] - 15)**2 + (x[15] - 16)**2 + (x[16] - 17)**2 + (x[17] - 18)**2

def f21(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2 + (x[2] - 3)**2 + (x[3] - 4)**2 + (x[4] - 5)**2 + (x[5] - 6)**2 + (x[6] - 7)**2 + (x[7] - 8)**2 + (x[8] - 9)**2 + (x[9] - 10)**2 + (x[10] - 11)**2 + (x[11] - 12)**2 + (x[12] - 13)**2 + (x[13] - 14)**2 + (x[14] - 15)**2 + (x[15] - 16)**2 + (x[16] - 17)**2 + (x[17] - 18)**2 + (x[18] - 19)**2

def f22(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2 + (x[2] - 3)**2 + (x[3] - 4)**2 + (x[4] - 5)**2 + (x[5] - 6)**2 + (x[6] - 7)**2 + (x[7] - 8)**2 + (x[8] - 9)**2 + (x[9] - 10)**2 + (x[10] - 11)**2 + (x[11] - 12)**2 + (x[12] - 13)**2 + (x[13] - 14)**2 + (x[14] - 15)**2 + (x[15] - 16)**2 + (x[16] - 17)**2 + (x[17] - 18)**2 + (x[18] - 19)**2 + (x[19] - 20)**2

def f23(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2 + (x[2] - 3)**2 + (x[3] - 4)**2 + (x[4] - 5)**2 + (x[5] - 6)**2 + (x[6] - 7)**2 + (x[7] - 8)**2 + (x[8] - 9)**2 + (x[9] - 10)**2 + (x[10] - 11)**2 + (x[11] - 12)**2 + (x[12] - 13)**2 + (x[13] - 14)**2 + (x[14] - 15)**2 + (x[15] - 16)**2 + (x[16] - 17)**2 + (x[17] - 18)**2 + (x[18] - 19)**2 + (x[19] - 20)**2 + (x[20] - 21)**2

def f24(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2 + (x[2] - 3)**2 + (x[3] - 4)**2 + (x[4] - 5)**2 + (x[5] - 6)**2 + (x[6] - 7)**2 + (x[7] - 8)**2 + (x[8] - 9)**2 + (x[9] - 10)**2 + (x[10] - 11)**2 + (x[11] - 12)**2 + (x[12] - 13)**2 + (x[13] - 14)**2 + (x[14] - 15)**2 + (x[15] - 16)**2 + (x[16] - 17)**2 + (x[17] - 18)**2 + (x[18] - 19)**2 + (x[19] - 20)**2 + (x[20] - 21)**2 + (x[21] - 22)**2

# Usage
algorithm = HybridEvolutionaryAlgorithm(100, 2)
algorithm.optimize(f1)