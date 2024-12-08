import numpy as np

class QuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(4 * dim, 25)  # Slightly increased minimal population size
        self.mutation_factor = 0.65  # Adjusted for fine balance
        self.crossover_rate = 0.88  # Adjusted for better exploration
        self.quantum_intensity = 4  # Introduced for quantum exploration
        self.adaptive_threshold = 0.25  # Adjusted for sensitivity to convergence
        self.population = np.random.uniform(-5, 5, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.used_budget = 0

    def quantum_parallelism(self, func):
        for i in range(self.quantum_intensity):
            if self.used_budget >= self.budget:
                break
            quantum_samples = np.random.uniform(-5, 5, (self.population_size, self.dim))
            quantum_fitness = np.array([func(ind) for ind in quantum_samples])
            self.used_budget += self.population_size
            better_indices = quantum_fitness < self.fitness
            self.fitness = np.where(better_indices, quantum_fitness, self.fitness)
            self.population = np.where(better_indices[:, None], quantum_samples, self.population)

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
        best_indices = np.argsort(self.fitness)[:self.quantum_intensity]
        for idx in best_indices:
            if self.used_budget >= self.budget:
                break
            step_size = np.random.uniform(0.03, 0.15)  # Adjusted for enhanced adaptability
            perturbation = np.random.uniform(-step_size, step_size, self.dim)
            candidate = np.clip(self.population[idx] + perturbation, -5, 5)
            candidate_fitness = func(candidate)
            self.used_budget += 1
            if candidate_fitness < self.fitness[idx]:
                self.fitness[idx] = candidate_fitness
                self.population[idx] = candidate

    def adapt_parameters(self):
        if np.std(self.fitness) < self.adaptive_threshold:
            self.quantum_intensity = 6  # Increased to enhance exploratory capabilities
            self.mutation_factor = 0.75  # Adjusted for improved diversity
            self.crossover_rate = 0.9  # Slightly refined for balancing exploration

    def __call__(self, func):
        self.fitness = np.array([func(ind) for ind in self.population])
        self.used_budget = self.population_size
        while self.used_budget < self.budget:
            self.quantum_parallelism(func)
            self.differential_evolution(func)
            self.stochastic_local_search(func)
            self.adapt_parameters()
        best_index = np.argmin(self.fitness)
        return self.population[best_index]