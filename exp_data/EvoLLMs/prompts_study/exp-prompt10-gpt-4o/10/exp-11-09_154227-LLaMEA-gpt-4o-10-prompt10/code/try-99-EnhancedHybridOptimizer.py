import numpy as np

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 22  # Slightly increased for broader coverage
        self.initial_scale_factor = 0.9  # Enhanced mutation aggressiveness
        self.initial_crossover_prob = 0.8  # Increased diversity through crossover
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = self.opposition_based_initialization()
        self.fitness = np.full(self.population_size, np.inf)
        self.evaluations = 0
        self.learning_rate = 0.15  # Enhanced local search efficacy
        self.elitism_rate = 0.15  # More elites preserved for robustness

    def opposition_based_initialization(self):
        initial_population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        opposition_population = self.lower_bound + self.upper_bound - initial_population
        combined_population = np.vstack((initial_population, opposition_population))
        fitness = np.array([np.inf] * len(combined_population))
        for i in range(len(combined_population)):
            fitness[i] = np.random.uniform(0.5, 1.5)  # Random scaling for initial diversity
        best_indices = np.argsort(fitness)[:self.population_size]
        return combined_population[best_indices]

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

    def elitism_selection(self):
        num_elites = max(1, int(self.elitism_rate * self.population_size))
        elite_indices = np.argsort(self.fitness)[:num_elites]
        return self.population[elite_indices], self.fitness[elite_indices]

    def __call__(self, func):
        self.evaluate_population(func)
        while self.evaluations < self.budget:
            elites, elite_fitness = self.elitism_selection()
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                scale_factor = self.initial_scale_factor * (1 - self.evaluations / self.budget)
                crossover_prob = self.initial_crossover_prob + 0.10 * np.random.rand()  # Slightly reduced randomness

                if i < len(elites):  # Use elite individuals
                    candidate_vector = self.local_search(elites[i])
                else:
                    mutant_vector = self.differential_evolution_mutation(i, scale_factor)
                    candidate_vector = self.crossover(self.population[i], mutant_vector, crossover_prob)

                candidate_fitness = func(candidate_vector)
                self.evaluations += 1

                if candidate_fitness < self.fitness[i]:
                    self.population[i] = candidate_vector
                    self.fitness[i] = candidate_fitness

                self.learning_rate = max(0.01, self.learning_rate * (1 - 0.03 * np.random.rand()))  # Slightly adjusted decay

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]