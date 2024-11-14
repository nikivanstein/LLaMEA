import numpy as np

class HybridStochasticOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 25  # Increased for better diversity
        self.initial_scale_factor = 0.9  # Slightly higher for more exploration
        self.initial_crossover_prob = 0.75  # More stringent crossover probability
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.evaluations = 0

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

    def gradient_inspired_local_search(self, vector, func):
        gradient_step = np.random.normal(0, 0.05, self.dim)  # Smaller step for fine-tuning
        perturbed_vector = vector + gradient_step
        current_fitness = func(vector)
        perturbed_fitness = func(perturbed_vector)
        self.evaluations += 2
        if perturbed_fitness < current_fitness:
            return np.clip(perturbed_vector, self.lower_bound, self.upper_bound)
        return vector

    def __call__(self, func):
        self.evaluate_population(func)
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                # Dynamic adjustment of parameters
                scale_factor = self.initial_scale_factor * (1 - self.evaluations / self.budget)
                crossover_prob = self.initial_crossover_prob + 0.1 * np.random.rand()  # Reduced randomness

                mutant_vector = self.differential_evolution_mutation(i, scale_factor)
                trial_vector = self.crossover(self.population[i], mutant_vector, crossover_prob)
                trial_fitness = func(trial_vector)
                self.evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial_vector
                    self.fitness[i] = trial_fitness

                if np.random.rand() < 0.25:  # Slightly increased probability for local search
                    candidate_vector = self.gradient_inspired_local_search(self.population[i], func)
                    candidate_fitness = func(candidate_vector)
                    self.evaluations += 1
                    if candidate_fitness < self.fitness[i]:
                        self.population[i] = candidate_vector
                        self.fitness[i] = candidate_fitness

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]