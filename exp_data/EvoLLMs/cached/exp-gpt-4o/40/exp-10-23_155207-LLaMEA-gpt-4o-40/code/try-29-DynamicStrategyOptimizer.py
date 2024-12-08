import numpy as np

class DynamicStrategyOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50  # Increased slightly for better diversity
        self.mutation_factor = 0.9  # Enhanced for stronger exploration
        self.crossover_probability = 0.8  # Adjusted for balance in recombination
        self.learning_rate = 0.05  # Reduced for fine-tuned local search
        self.population = None
        self.fitness = None

    def initialize_population(self):
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, 
                                            (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])

    def adaptive_differential_evolution(self, idx, func):
        indices = [i for i in range(self.population_size) if i != idx]
        np.random.shuffle(indices)
        a, b, c = indices[:3]
        mutant = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
        adaptive_factor = 0.6 + 0.4 * np.random.rand()  # Adjusted range for exploration
        mutant = self.population[idx] + adaptive_factor * (mutant - self.population[idx])
        cross_points = np.random.rand(self.dim) < self.crossover_probability
        if not np.any(cross_points):
            index = np.random.randint(0, self.dim)
            cross_points[index] = True
        
        trial = np.where(cross_points, mutant, self.population[idx])
        trial = np.clip(trial, self.lower_bound, self.upper_bound)
        trial_fitness = func(trial)
        
        if trial_fitness < self.fitness[idx]:
            self.population[idx] = trial
            self.fitness[idx] = trial_fitness

    def variational_stochastic_update(self, idx, func):
        gradient = np.random.uniform(-2, 2, self.dim)  # Expanded to allow more diverse searches
        adaptive_lr = self.learning_rate / (1 + idx / self.population_size)
        hessian_approx = np.random.uniform(0.2, 0.6, self.dim)  # Broader range for more variability
        candidate = self.population[idx] - adaptive_lr * gradient / (np.abs(hessian_approx) + 1e-9)
        candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
        candidate_fitness = func(candidate)
        if candidate_fitness < self.fitness[idx]:
            self.population[idx] = candidate
            self.fitness[idx] = candidate_fitness

    def __call__(self, func):
        self.initialize_population()
        evaluations = 0
        self.evaluate_population(func)
        evaluations += self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                self.adaptive_differential_evolution(i, func)
                evaluations += 1

                if evaluations >= self.budget:
                    break

                self.variational_stochastic_update(i, func)
                evaluations += 1

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]