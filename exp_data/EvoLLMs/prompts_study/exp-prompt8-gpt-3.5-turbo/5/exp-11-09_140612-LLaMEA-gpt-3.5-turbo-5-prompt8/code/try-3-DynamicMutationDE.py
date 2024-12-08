class DynamicMutationDE(FastConvergingDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

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
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                for j in range(self.dim):
                    if np.random.rand() > CR:
                        mutant[j] = pop[i][j]
                new_fit = func(mutant)
                if new_fit < fitness[i]:
                    pop[i] = mutant
                    fitness[i] = new_fit
                
                # Dynamic mutation strategy - adaptively adjust F based on performance
                if new_fit < fitness[i] and np.random.rand() < 0.5:
                    F = max(0.1, min(0.9, F + 0.05))
        
        best_idx = np.argmin(fitness)
        best_solution = pop[best_idx]
        return best_solution