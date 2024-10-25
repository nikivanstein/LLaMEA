import numpy as np

class QuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5 * dim, 25)  # Increased minimal population size for diversity
        self.mutation_factor = 0.6  # Slightly reduced for controlled diversity
        self.crossover_rate = 0.85  # Reduced to maintain stable convergence
        self.local_search_intensity = 6  # Balanced for effective local exploration
        self.adaptive_threshold = 0.15  # Lowered for sensitivity to change
        self.population = np.random.uniform(-5, 5, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.used_budget = 0

    def quantum_differential_evolution(self, func):
        for i in range(self.population_size):
            if self.used_budget >= self.budget:
                break
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
            mutant_vector = np.clip(a + self.mutation_factor * (b - c), -5, 5)
            quantum_influence = np.random.uniform(-0.1, 0.1, self.dim)  # Quantum-inspired influence
            crossover = np.random.rand(self.dim) < self.crossover_rate
            trial_vector = np.where(crossover, mutant_vector + quantum_influence, self.population[i])
            trial_fitness = func(trial_vector)
            self.used_budget += 1
            if trial_fitness < self.fitness[i]:
                self.fitness[i] = trial_fitness
                self.population[i] = trial_vector

    def enhanced_local_search(self, func):
        best_indices = np.argsort(self.fitness)[:self.local_search_intensity]
        for idx in best_indices:
            if self.used_budget >= self.budget:
                break
            step_size = np.random.uniform(0.03, 0.15)  # Adjusted for finer control in local search
            perturbation = np.random.uniform(-step_size, step_size, self.dim)
            candidate = np.clip(self.population[idx] + perturbation, -5, 5)
            candidate_fitness = func(candidate)
            self.used_budget += 1
            if candidate_fitness < self.fitness[idx]:
                self.fitness[idx] = candidate_fitness
                self.population[idx] = candidate

    def adapt_parameters(self):
        if np.std(self.fitness) < self.adaptive_threshold or self.used_budget > self.budget * 0.8:
            self.local_search_intensity = 8  # Intensified in late stages of the search
            self.mutation_factor = 0.75  # Adjusted for diversity control
            self.crossover_rate = 0.9  # Fine-tuned for better recombination

    def __call__(self, func):
        self.fitness = np.array([func(ind) for ind in self.population])
        self.used_budget = self.population_size
        while self.used_budget < self.budget:
            self.quantum_differential_evolution(func)
            self.enhanced_local_search(func)
            self.adapt_parameters()
        best_index = np.argmin(self.fitness)
        return self.population[best_index]