from numpy.random import uniform, normal

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        pop_size = 10
        c1, c2 = 1.496, 1.496
        w = 0.5
        p_best = uniform(-5.0, 5.0, (pop_size, self.dim))
        v = normal(0, 1, (pop_size, self.dim))
        
        for _ in range(self.budget):
            r1, r2 = uniform(0, 1, (pop_size, self.dim)), uniform(0, 1, (pop_size, self.dim))
            v = w*v + c1*r1*(p_best - X) + c2*r2*(gbest - X)
            X = X + v
            X = np.clip(X, -5.0, 5.0)
            if func(X) < func(p_best):
                p_best = X
        return p_best