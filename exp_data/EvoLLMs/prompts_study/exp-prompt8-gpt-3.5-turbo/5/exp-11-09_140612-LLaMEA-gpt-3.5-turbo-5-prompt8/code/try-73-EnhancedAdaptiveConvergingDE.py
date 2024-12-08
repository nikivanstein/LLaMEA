import numpy as np

class EnhancedAdaptiveConvergingDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        pop_size = 10
        pop = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        CR = np.full(pop_size, 0.5)  # Dynamic adjustment of crossover rate
        F = np.full(pop_size, 0.5)   # Dynamic adjustment of mutation factor
        
        for _ in range(self.budget):
            new_pop = np.copy(pop)
            for i in range(pop_size):
                candidates = np.random.choice(pop_size, size=5, replace=False)  # Increasing candidate pool size
                r1, r2, r3, r4, r5 = candidates
                mutant = pop[r1] + F[i] * (pop[r2] - pop[r3]) + F[i] * (pop[r4] - pop[r5])  # Utilizing additional vectors for mutation
                for j in range(self.dim):
                    if np.random.rand() > CR[i]:
                        mutant[j] = pop[i][j]
                new_fit = func(mutant)
                if new_fit < fitness[i]:
                    pop[i] = mutant
                    fitness[i] = new_fit
                    # Adapt mutation and crossover rates based on population diversity
                    if np.mean(fitness) != 0:  # Dynamic adaptation based on population diversity
                        CR[i] = np.clip(CR[i] + np.random.normal(0, 0.1), 0, 1)
                        F[i] = np.clip(F[i] + np.random.normal(0, 0.1), 0, 2)
            
            if np.random.rand() < 0.1:  # Adjust population size based on population diversity
                avg_fitness = np.mean(fitness)
                if avg_fitness < np.median(fitness) and pop_size < 20:
                    new_member = np.random.uniform(-5.0, 5.0, (1, self.dim))
                    pop = np.append(pop, new_member, axis=0)
                    fitness = np.append(fitness, func(new_member))
                    CR = np.append(CR, np.random.uniform(0, 1))
                    F = np.append(F, np.random.uniform(0, 2))
                elif avg_fitness > np.median(fitness) and pop_size > 5:
                    worst_idx = np.argmax(fitness)
                    pop = np.delete(pop, worst_idx, axis=0)
                    fitness = np.delete(fitness, worst_idx)
                    CR = np.delete(CR, worst_idx)
                    F = np.delete(F, worst_idx)
            
            pop_size = len(pop)
                    
        best_idx = np.argmin(fitness)
        best_solution = pop[best_idx]
        return best_solution