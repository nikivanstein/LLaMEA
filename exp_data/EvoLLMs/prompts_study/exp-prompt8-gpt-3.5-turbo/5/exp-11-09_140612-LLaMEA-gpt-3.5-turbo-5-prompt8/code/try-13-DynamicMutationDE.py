import numpy as np

class DynamicMutationDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        CR = 0.5
        F = 0.5
        pop_size = 10
        pop = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        
        for _ in range(self.budget):
            new_pop = np.copy(pop)
            for i in range(pop_size):
                candidates = np.random.choice(pop_size, size=3, replace=False)
                r1, r2, r3 = candidates
                
                # Dynamic mutation adaptation
                F = 0.5 if np.mean(fitness) > np.median(fitness) else 0.8
                CR = 0.5 if np.std(fitness) < np.mean(fitness) else 0.9
                
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                for j in range(self.dim):
                    if np.random.rand() > CR:
                        mutant[j] = pop[i][j]
                new_fit = func(mutant)
                if new_fit < fitness[i]:
                    pop[i] = mutant
                    fitness[i] = new_fit
                    
        best_idx = np.argmin(fitness)
        best_solution = pop[best_idx]
        return best_solution