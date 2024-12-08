import numpy as np

class CuckooDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_cuckoos = 20
        self.num_nests = 5
        self.pa = 0.25

    def __call__(self, func):
        def levy_flight(scale=1.0):
            beta = 1.5
            sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
            u = np.random.normal(0, sigma, self.dim)
            v = np.random.normal(0, 1, self.dim)
            step = u / np.abs(v)**(1 / beta)
            step *= scale
            return step

        def de(func):
            pop = np.random.uniform(-5.0, 5.0, (self.num_nests, self.dim))
            fitness = np.array([func(ind) for ind in pop])
            best_idx = np.argmin(fitness)
            best = pop[best_idx]
            
            for _ in range(self.budget):
                new_pop = np.zeros((self.num_nests, self.dim))
                for i in range(self.num_nests):
                    idxs = np.arange(self.num_nests)
                    np.random.shuffle(idxs)
                    r1, r2, r3 = pop[idxs[:3]]
                    mutant = r1 + 0.5 * (r2 - r3)
                    mutant = np.clip(mutant, -5.0, 5.0)
                    cross_points = np.random.rand(self.dim) < self.pa
                    trial = np.where(cross_points, mutant, pop[i])
                    if func(trial) < fitness[i]:
                        pop[i] = trial
                        fitness[i] = func(trial)
                    if fitness[i] < func(best):
                        best = pop[i]

            return best

        cuckoos = np.random.uniform(-5.0, 5.0, (self.num_cuckoos, self.dim))
        best_cuckoo = cuckoos[np.argmin([func(cuck) for cuck in cuckoos])]
        
        for _ in range(self.budget // self.num_cuckoos):
            step_size = levy_flight()
            for i in range(self.num_cuckoos):
                cuckoo = cuckoos[i] + step_size
                cuckoo = np.clip(cuckoo, -5.0, 5.0)
                if func(cuckoo) < func(best_cuckoo):
                    best_cuckoo = cuckoo

        return de(func)
        