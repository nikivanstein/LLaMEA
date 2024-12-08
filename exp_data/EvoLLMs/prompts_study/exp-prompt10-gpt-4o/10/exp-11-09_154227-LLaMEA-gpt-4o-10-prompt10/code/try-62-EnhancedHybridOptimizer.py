import numpy as np

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.initial_scale_factor = 0.9  # Slightly increased for faster exploration
        self.initial_crossover_prob = 0.8  # Slightly increased to maintain diversity
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.opposition_rate = 0.1  # New parameter for dynamic opposition
        self.population = self.opposition_based_initialization()
        self.fitness = np.full(self.population_size, np.inf)
        self.evaluations = 0
        self.learning_rate = 0.1

    def opposition_based_initialization(self):
        initial_population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        opposition_population = self.lower_bound + self.upper_bound - initial_population
        return self.multi_level_opposition(initial_population, opposition_population)
    
    def multi_level_opposition(self, population, opposition_population):
        combined_population = np.vstack((population, opposition_population))
        multi_opposition_population = self.lower_bound + self.upper_bound - combined_population
        all_population = np.vstack((combined_population, multi_opposition_population))
        return all_population[:self.population_size]

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

                # Dynamic adjustment of parameters
                scale_factor = self.initial_scale_factor * np.random.uniform(0.8, 1.2)  # Added randomness
                crossover_prob = self.initial_crossover_prob * np.random.uniform(0.8, 1.2) 

                mutant_vector = self.differential_evolution_mutation(i, scale_factor)
                trial_vector = self.crossover(self.population[i], mutant_vector, crossover_prob)
                trial_fitness = func(trial_vector)
                self.evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial_vector
                    self.fitness[i] = trial_fitness

                if np.random.rand() < self.opposition_rate:  # Conditional opposition-based learning
                    candidate_vector = self.opposition_based_local_search(self.population[i])
                    candidate_fitness = func(candidate_vector)
                    self.evaluations += 1
                    if candidate_fitness < self.fitness[i]:
                        self.population[i] = candidate_vector
                        self.fitness[i] = candidate_fitness

                # Update learning rate adaptively
                self.learning_rate = max(0.01, self.learning_rate * (1 - 0.05 * np.random.rand()))

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]

    def opposition_based_local_search(self, vector):
        opposite_vector = self.lower_bound + self.upper_bound - vector
        perturbation = np.random.normal(0, self.learning_rate, self.dim)
        candidate_vector = opposite_vector + perturbation
        return np.clip(candidate_vector, self.lower_bound, self.upper_bound)