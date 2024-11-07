import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.mutation_factor = 0.5 + np.random.rand(self.population_size) * 0.5
        self.crossover_probability = 0.9

    def __call__(self, func):
        np.random.seed(42)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.asarray([func(ind) for ind in population])
        evaluations = self.population_size
        success_rate = np.zeros(self.population_size)
        
        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                F = self.mutation_factor[i]  # Use individual-specific mutation factor
                mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)

                adaptive_crossover = min(1.0, self.crossover_probability + 0.1 * (1 - success_rate[i]))
                cross_points = np.random.rand(self.dim) < adaptive_crossover

                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                f_trial = func(trial)
                evaluations += 1

                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    success_rate[i] = 1
                    self.mutation_factor[i] = min(1.0, self.mutation_factor[i] + 0.1)  # Increase F upon success
                else:
                    success_rate[i] = 0.9 * success_rate[i]
                    self.mutation_factor[i] = max(0.5, self.mutation_factor[i] - 0.1)  # Decrease F upon failure

                if f_trial < best_fitness:
                    best_solution = trial
                    best_fitness = f_trial

            # Elitism: ensure best solution is retained
            population[np.argmax(fitness)] = best_solution
            fitness[np.argmax(fitness)] = best_fitness

        return best_solution