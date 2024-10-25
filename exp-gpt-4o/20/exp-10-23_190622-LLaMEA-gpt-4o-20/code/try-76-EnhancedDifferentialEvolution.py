import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5 * dim, 25)  # Slightly increased population size
        self.mutation_factor = 0.7
        self.crossover_rate = 0.9
        self.local_search_intensity = 5
        self.adaptive_threshold = 0.15  # Fine-tuned adaptive threshold
        self.population = np.random.uniform(-5, 5, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.used_budget = 0
        self.surrogate_model = GaussianProcessRegressor(kernel=RBF(), n_restarts_optimizer=5)  # Surrogate model

    def differential_evolution(self, func):
        for i in range(self.population_size):
            if self.used_budget >= self.budget:
                break
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
            mutant_vector = np.clip(a + self.mutation_factor * (b - c), -5, 5)
            crossover = np.random.rand(self.dim) < self.crossover_rate
            trial_vector = np.where(crossover, mutant_vector, self.population[i])
            trial_fitness = func(trial_vector)
            self.used_budget += 1
            if trial_fitness < self.fitness[i]:
                self.fitness[i] = trial_fitness
                self.population[i] = trial_vector

    def stochastic_local_search(self, func):
        best_indices = np.argsort(self.fitness)[:self.local_search_intensity]
        for idx in best_indices:
            if self.used_budget >= self.budget:
                break
            step_size = np.random.uniform(0.05, 0.25)  # Broadened step size for adaptability
            perturbation = np.random.uniform(-step_size, step_size, self.dim)
            candidate = np.clip(self.population[idx] + perturbation, -5, 5)
            candidate_fitness = func(candidate)
            self.used_budget += 1
            if candidate_fitness < self.fitness[idx]:
                self.fitness[idx] = candidate_fitness
                self.population[idx] = candidate

    def surrogate_assisted_search(self, func):
        if self.used_budget >= self.budget:
            return
        training_data = np.array([ind for ind in self.population])
        training_targets = np.array(self.fitness)
        self.surrogate_model.fit(training_data, training_targets)
        random_samples = np.random.uniform(-5, 5, (100, self.dim))
        predicted_fitness = self.surrogate_model.predict(random_samples)
        best_pred_idx = np.argmin(predicted_fitness)
        candidate = random_samples[best_pred_idx]
        candidate_fitness = func(candidate)
        self.used_budget += 1
        if candidate_fitness < np.max(self.fitness):
            worst_idx = np.argmax(self.fitness)
            self.population[worst_idx] = candidate
            self.fitness[worst_idx] = candidate_fitness

    def adapt_parameters(self):
        if np.std(self.fitness) < self.adaptive_threshold:
            self.local_search_intensity = 6
            self.mutation_factor = 0.75
            self.crossover_rate = 0.91

    def __call__(self, func):
        self.fitness = np.array([func(ind) for ind in self.population])
        self.used_budget = self.population_size
        while self.used_budget < self.budget:
            self.differential_evolution(func)
            self.stochastic_local_search(func)
            self.surrogate_assisted_search(func)  # New surrogate search step
            self.adapt_parameters()
        best_index = np.argmin(self.fitness)
        return self.population[best_index]