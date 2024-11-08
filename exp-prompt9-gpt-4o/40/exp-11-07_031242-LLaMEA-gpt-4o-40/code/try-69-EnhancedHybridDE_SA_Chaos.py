import numpy as np

class EnhancedHybridDE_SA_Chaos:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb, self.ub = -5.0, 5.0
        self.pop_size = 30
        self.min_pop_size = 10
        self.mutation_factor = 0.7
        self.cr = 0.85
        self.temp_init = 120.0
        self.temp_decay = 0.9

    def __call__(self, func):
        np.random.seed(42)
        pop_size, lb, ub = self.pop_size, self.lb, self.ub
        population = np.random.uniform(lb, ub, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals_used = pop_size

        best_idx = np.argmin(fitness)
        best_solution, best_fitness = population[best_idx], fitness[best_idx]

        def de_mutation(target_idx):
            indices = list(range(pop_size))
            indices.remove(target_idx)
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)
            cross_points = np.random.rand(self.dim) < self.cr
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            return np.where(cross_points, mutant, population[target_idx])

        temperature = self.temp_init
        while evals_used < self.budget:
            for i in range(pop_size):
                trial = de_mutation(i)
                trial_fitness = func(trial)
                evals_used += 1

                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / temperature):
                    population[i], fitness[i] = trial, trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution, best_fitness = trial, trial_fitness

                temperature *= self.temp_decay

                if evals_used >= self.budget:
                    break

            pop_size = max(self.min_pop_size, int(self.pop_size * (1 - evals_used / self.budget)))
            population, fitness = population[:pop_size], fitness[:pop_size]

        return best_solution, best_fitness