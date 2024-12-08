import numpy as np

class DEALS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9

    def __call__(self, func):
        # Initialize population randomly
        population = self.lower_bound + np.random.rand(self.population_size, self.dim) * (self.upper_bound - self.lower_bound)
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations += self.population_size
        
        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                # Select indices for mutation
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)

                # Mutation
                mutant = np.clip(population[a] + self.mutation_factor * (population[b] - population[c]), 
                                 self.lower_bound, self.upper_bound)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.crossover_probability
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate trial vector
                trial_fitness = func(trial)
                self.evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update best solution
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                # Adaptive Local Search (ALS)
                if np.random.rand() < 0.1:  # 10% chance to explore locally
                    local_search_point = np.clip(trial + np.random.normal(0, 0.1, self.dim), 
                                                 self.lower_bound, self.upper_bound)
                    local_fitness = func(local_search_point)
                    self.evaluations += 1

                    if local_fitness < fitness[i]:
                        population[i] = local_search_point
                        fitness[i] = local_fitness

                        if local_fitness < best_fitness:
                            best_solution = local_search_point
                            best_fitness = local_fitness

                if self.evaluations >= self.budget:
                    break

        return best_solution, best_fitness