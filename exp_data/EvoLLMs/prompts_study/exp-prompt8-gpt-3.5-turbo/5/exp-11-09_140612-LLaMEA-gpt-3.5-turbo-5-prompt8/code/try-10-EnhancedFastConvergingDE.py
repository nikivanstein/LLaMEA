class EnhancedFastConvergingDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        CR = 0.5
        pop_size = 10
        pop = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        F_min, F_max = 0.2, 0.8

        for _ in range(self.budget):
            new_pop = np.copy(pop)
            for i in range(pop_size):
                candidates = np.random.choice(pop_size, size=3, replace=False)
                r1, r2, r3 = candidates
                F = np.clip(np.random.normal(0.5, 0.1), F_min, F_max)  # Dynamic F adaptation
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