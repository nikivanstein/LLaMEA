import numpy as np

class EnhancedHybridStochasticOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20  # Starting size
        self.scale_factor = 0.8
        self.crossover_prob = 0.7
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.evaluations = 0

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.fitness[i] == np.inf:  # If not evaluated yet
                self.fitness[i] = func(self.population[i])
                self.evaluations += 1

    def select_parents(self, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        return np.random.choice(indices, 3, replace=False)

    def differential_evolution_mutation(self, idx):
        a, b, c = self.select_parents(idx)
        mutant_vector = self.population[a] + self.scale_factor * (self.population[b] - self.population[c])
        return np.clip(mutant_vector, self.lower_bound, self.upper_bound)

    def crossover(self, target_vector, mutant_vector):
        crossover_mask = np.random.rand(self.dim) < self.adaptive_crossover_prob()
        return np.where(crossover_mask, mutant_vector, target_vector)

    def adaptive_crossover_prob(self):
        return 0.7 + 0.3 * np.random.rand()

    def local_search(self, vector):
        perturbation = np.random.normal(0, 0.1, self.dim)  # Small Gaussian perturbations
        candidate_vector = vector + perturbation
        return np.clip(candidate_vector, self.lower_bound, self.upper_bound)

    def dynamic_population_adjustment(self):
        if self.evaluations < self.budget * 0.5:
            return 20 + int(5 * (1 - (self.evaluations / self.budget)))
        else:
            return max(10, self.population_size - int(5 * ((self.evaluations - self.budget * 0.5) / (self.budget * 0.5))))

    def __call__(self, func):
        self.evaluate_population(func)
        while self.evaluations < self.budget:
            self.population_size = self.dynamic_population_adjustment()
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                mutant_vector = self.differential_evolution_mutation(i)
                trial_vector = self.crossover(self.population[i], mutant_vector)
                trial_fitness = func(trial_vector)
                self.evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial_vector
                    self.fitness[i] = trial_fitness

                if np.random.rand() < 0.2:  # Probability of local search
                    candidate_vector = self.local_search(self.population[i])
                    candidate_fitness = func(candidate_vector)
                    self.evaluations += 1
                    if candidate_fitness < self.fitness[i]:
                        self.population[i] = candidate_vector
                        self.fitness[i] = candidate_fitness

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]