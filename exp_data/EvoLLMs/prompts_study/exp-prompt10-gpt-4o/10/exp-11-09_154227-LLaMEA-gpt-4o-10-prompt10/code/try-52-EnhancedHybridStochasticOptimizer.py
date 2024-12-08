import numpy as np

class EnhancedHybridStochasticOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.initial_scale_factor = 0.9  # Adjusted for more exploration
        self.initial_crossover_prob = 0.75  # Slight increase for diversity
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = self.opposition_based_initialization()
        self.fitness = np.full(self.population_size, np.inf)
        self.evaluations = 0
        self.learning_rate = 0.1
        self.elite_fraction = 0.1  # New parameter for elitism

    def opposition_based_initialization(self):
        initial_population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        opposition_population = self.lower_bound + self.upper_bound - initial_population
        return np.vstack((initial_population, opposition_population))[:self.population_size]

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
                self.evaluations += 1

    def select_parents(self, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        return np.random.choice(indices, 3, replace=False)

    def differential_evolution_mutation(self, idx, scale_factor):
        a, b, c = self.select_parents(idx)
        mutant_vector = self.population[a] + scale_factor * (self.population[b] - self.population[c])
        return np.clip(mutant_vector, self.lower_bound, self.upper_bound)

    def crossover(self, target_vector, mutant_vector, crossover_prob):
        crossover_mask = np.random.rand(self.dim) < crossover_prob
        return np.where(crossover_mask, mutant_vector, target_vector)

    def local_search(self, vector):
        perturbation = np.random.normal(0, self.learning_rate, self.dim)
        candidate_vector = vector + perturbation
        return np.clip(candidate_vector, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        self.evaluate_population(func)
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                # Dynamic adjustment of parameters with adaptive scaling
                scale_factor = self.initial_scale_factor * (1 - self.evaluations / self.budget) * (0.5 + 0.5 * np.random.rand())
                crossover_prob = self.initial_crossover_prob + 0.1 * np.random.rand()

                mutant_vector = self.differential_evolution_mutation(i, scale_factor)
                trial_vector = self.crossover(self.population[i], mutant_vector, crossover_prob)
                trial_fitness = func(trial_vector)
                self.evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial_vector
                    self.fitness[i] = trial_fitness

                if np.random.rand() < 0.2:
                    candidate_vector = self.local_search(self.population[i])
                    candidate_fitness = func(candidate_vector)
                    self.evaluations += 1
                    if candidate_fitness < self.fitness[i]:
                        self.population[i] = candidate_vector
                        self.fitness[i] = candidate_fitness

                # Update learning rate adaptively
                self.learning_rate = max(0.01, self.learning_rate * (1 - 0.05 * np.random.rand()))

            # Elitist selection to retain a fraction of the best solutions
            num_elite = int(self.elite_fraction * self.population_size)
            elite_indices = np.argsort(self.fitness)[:num_elite]
            self.population[:num_elite] = self.population[elite_indices]
            self.fitness[:num_elite] = self.fitness[elite_indices]

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]