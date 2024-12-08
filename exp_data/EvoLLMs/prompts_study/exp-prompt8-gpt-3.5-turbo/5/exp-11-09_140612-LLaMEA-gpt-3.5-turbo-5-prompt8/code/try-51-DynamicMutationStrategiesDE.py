import numpy as np

class DynamicMutationStrategiesDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        pop_size = 10
        pop = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        CR = np.full(pop_size, 0.5)  
        F = np.full(pop_size, 0.5)   

        for _ in range(self.budget):
            new_pop = np.copy(pop)
            for i in range(pop_size):
                candidates = np.random.choice(pop_size, size=3, replace=False)
                r1, r2, r3 = candidates
                mutant = pop[r1] + F[i] * (pop[r2] - pop[r3])
                for j in range(self.dim):
                    if np.random.rand() > CR[i]:
                        mutant[j] = pop[i][j]
                new_fit = func(mutant)
                if new_fit < fitness[i]:
                    pop[i] = mutant
                    fitness[i] = new_fit
                    # Dynamic mutation strategy selection based on individual performance
                    if new_fit < np.mean(fitness): 
                        F[i] = np.clip(F[i] + np.random.normal(0, 0.1), 0, 2)
                    else:
                        F[i] = np.clip(F[i] - np.random.normal(0, 0.1), 0, 2)
            
            best_idx = np.argmin(fitness)
            best_solution = pop[best_idx]
            
        return best_solution