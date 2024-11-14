import numpy as np

class EnhancedDynamicDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.init_pop_size = 40
        self.min_pop_size = 8
        self.mutation_factor = 0.8
        self.cr = 0.9  # Crossover rate
        self.initial_temp = 100.0
        self.temp_decay = 0.92

    def __call__(self, func):
        np.random.seed(0)
        pop_size = self.init_pop_size
        population = np.random.uniform(self.lower_bound, self.upper_bound, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals_used = pop_size

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        def de_mutation_and_crossover(target_idx, temp):
            indices = list(range(pop_size))
            indices.remove(target_idx)
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
            cross_points = np.random.rand(self.dim) < self.cr
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[target_idx])
            return trial

        temperature = self.initial_temp
        while evals_used < self.budget:
            for i in range(pop_size):
                trial = de_mutation_and_crossover(i, temperature)
                trial_fitness = func(trial)
                evals_used += 1

                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                temperature *= self.temp_decay

                if evals_used >= self.budget:
                    break

            # Adjust population size for better convergence
            pop_size = max(self.min_pop_size, int(self.init_pop_size * (1 - evals_used / self.budget)))
            population = population[:pop_size]
            fitness = fitness[:pop_size]

        return best_solution, best_fitness