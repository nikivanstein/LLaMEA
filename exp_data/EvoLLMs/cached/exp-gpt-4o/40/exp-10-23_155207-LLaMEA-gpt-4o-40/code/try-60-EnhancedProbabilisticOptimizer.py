import numpy as np

class EnhancedProbabilisticOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 40
        self.mutation_factor = 0.85
        self.crossover_probability = 0.95
        self.learning_rate = 0.2
        self.diversity_factor = 0.1
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
        adaptive_factor = 0.4 + 0.6 * np.random.rand()
        mutant = self.population[idx] + adaptive_factor * (mutant - self.population[idx])
        mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
        
        cross_points = np.random.rand(self.dim) < self.crossover_probability
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        
        trial = np.where(cross_points, mutant, self.population[idx])
        trial_fitness = func(trial)
        
        if trial_fitness < self.fitness[idx]:
            self.population[idx] = trial
            self.fitness[idx] = trial_fitness

    def stochastic_gradient_approx(self, idx, func):
        gradient = np.random.normal(0, 1, self.dim)
        adaptive_lr = self.learning_rate / (1 + idx / self.population_size)
        candidate = self.population[idx] - adaptive_lr * gradient
        candidate += self.diversity_factor * np.random.normal(0, 1, self.dim)
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

                self.stochastic_gradient_approx(i, func)
                evaluations += 1

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]