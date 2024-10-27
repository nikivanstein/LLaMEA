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

    def hybrid_evolution(self, func):
        for i in range(self.budget):
            if random.random() < 0.4:
                new_individual = random.uniform(self.bounds[0][0], self.bounds[0][1])
            else:
                new_individual = self.x0
            updated_individual = self.f(new_individual, self.logger)
            self.x0 = updated_individual

    def f(self, individual, logger):
        # Differential Evolution
        def de(func, bounds, x0, tol=1e-5, x0_init=None, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, scaling='uniform', strategy_init=None):
            from scipy.optimize import differential_evolution

            if x0_init is None:
                x0 = np.random.uniform(bounds[0][0], bounds[0][1], len(bounds))
            else:
                x0 = x0_init

            if scaling == 'uniform':
                scaling = lambda x: 1.0 / (1.0 + np.exp(-x))
            elif scaling == 'log-uniform':
                scaling = lambda x: np.log(1.0 + np.exp(-x))

            bounds = [(a, b) for a, b in zip(bounds, [scaling(a) for a in x0])]
            res = differential_evolution(func, bounds, x0=x0, tol=tol, strategy=strategy, cxpb=cxpb, mu=mu, w=w, scaling=scaling, strategy_init=strategy_init)

            return res.x

        # Hybrid Evolutionary Algorithm
        def hybrid(func, bounds, x0, tol=1e-5, x0_init=None, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, scaling='uniform', strategy_init=None):
            from scipy.optimize import differential_evolution

            if x0_init is None:
                x0 = np.random.uniform(bounds[0][0], bounds[0][1], len(bounds))
            else:
                x0 = x0_init

            bounds = [(a, b) for a, b in zip(bounds, [scaling(a) for a in x0])]

            def de_func(x):
                return func(x)

            res = differential_evolution(de_func, bounds, x0=x0, tol=tol, strategy=strategy, cxpb=cxpb, mu=mu, w=w, scaling=scaling, strategy_init=strategy_init)

            return res.x

        res = de(func, self.bounds, x0=self.x0, tol=tol, x0_init=x0_init, strategy=strategy, cxpb=cxpb, mu=mu, w=w, scaling=scaling, strategy_init=strategy_init)
        return hybrid(func, self.bounds, res, tol=tol, x0_init=x0_init, strategy=strategy, cxpb=cxpb, mu=mu, w=w, scaling=scaling, strategy_init=strategy_init)

# Usage
if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    budget = 50
    dim = 10

    heacombbo = HybridEvolutionaryAlgorithm(budget, dim)

    # Test the algorithm
    def f(x):
        return sum([i**2 for i in x])

    heacombbo.logger = lambda x: x
    heacombbo.f = f

    for _ in range(100):
        heacombbo.hybrid_evolution(f)

    # Print the optimized value
    print("Optimized value:", f(heacombbo.x0))

    # Plot the objective function
    x = np.linspace(-5, 5, 100)
    y = [f(i) for i in x]
    plt.plot(x, y)
    plt.scatter(heacombbo.x0, f(heacombbo.x0), color='r')
    plt.show()