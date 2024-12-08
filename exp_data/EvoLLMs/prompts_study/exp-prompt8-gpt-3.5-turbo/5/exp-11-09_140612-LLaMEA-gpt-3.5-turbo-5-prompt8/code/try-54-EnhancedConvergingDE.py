import numpy as np

class EnhancedConvergingDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        pop_size = 10
        pop = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        CR = np.full(pop_size, 0.5)  # Dynamic adjustment of crossover rate
        F = np.full(pop_size, 0.5)   # Dynamic adjustment of mutation factor
        exploration_factor = 0.1  # Novel exploration factor
        
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
                    # Adapt mutation and crossover rates based on individual performance
                    if np.random.rand() < 0.1:  # Adjust rates with a probability
                        CR[i] = np.clip(CR[i] + np.random.normal(0, 0.1), 0, 1)
                        F[i] = np.clip(F[i] + np.random.normal(0, 0.1), 0, 2)
                
                # Encourage exploration in the search space
                for j in range(self.dim):
                    if np.random.rand() < exploration_factor:
                        pop[i][j] = np.random.uniform(-5.0, 5.0)
            
            # Adjust population size based on performance
            if np.random.rand() < 0.1:  # With a probability, add or remove individuals
                if np.random.rand() < 0.5 and pop_size < 20:  # Add individual
                    new_member = np.random.uniform(-5.0, 5.0, (1, self.dim))
                    pop = np.append(pop, new_member, axis=0)
                    fitness = np.append(fitness, func(new_member))
                    CR = np.append(CR, np.random.uniform(0, 1))
                    F = np.append(F, np.random.uniform(0, 2))
                elif pop_size > 5:  # Remove individual
                    worst_idx = np.argmax(fitness)
                    pop = np.delete(pop, worst_idx, axis=0)
                    fitness = np.delete(fitness, worst_idx)
                    CR = np.delete(CR, worst_idx)
                    F = np.delete(F, worst_idx)
            
            pop_size = len(pop)
                    
        best_idx = np.argmin(fitness)
        best_solution = pop[best_idx]
        return best_solution