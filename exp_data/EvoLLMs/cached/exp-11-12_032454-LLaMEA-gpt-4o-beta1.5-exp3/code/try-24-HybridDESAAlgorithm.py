import numpy as np

class HybridDESAAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(5, int(budget / (10 * dim)))  # heuristic for population size
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.initial_temperature = 1000
        self.cooling_rate = 0.99
        self.temperature = self.initial_temperature

    def __call__(self, func):
        # Initialize the population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size
        
        global_best_index = np.argmin(fitness)
        global_best = population[global_best_index]
        global_best_fitness = fitness[global_best_index]

        while num_evaluations < self.budget:
            # Differential Evolution Mutation and Crossover
            new_population = np.copy(population)
            for i in range(self.population_size):
                if num_evaluations >= self.budget:
                    break
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant_vector = np.clip(a + self.mutation_factor * (b - c), self.lb, self.ub)
                crossover = np.random.rand(self.dim) < self.crossover_probability
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial_vector = np.where(crossover, mutant_vector, population[i])
                trial_fitness = func(trial_vector)
                num_evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    new_population[i] = trial_vector
                    fitness[i] = trial_fitness
                    if trial_fitness < global_best_fitness:
                        global_best = trial_vector
                        global_best_fitness = trial_fitness
                else:
                    # Simulated Annealing acceptance
                    if np.random.rand() < np.exp(-(trial_fitness - fitness[i]) / self.temperature):
                        new_population[i] = trial_vector
                        fitness[i] = trial_fitness

            population = new_population
            self.temperature *= self.cooling_rate

        return global_best, global_best_fitness