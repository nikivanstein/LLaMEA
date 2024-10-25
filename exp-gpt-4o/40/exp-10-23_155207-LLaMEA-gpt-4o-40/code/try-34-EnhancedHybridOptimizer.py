import numpy as np

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 40
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.learning_rate = 0.1
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
        if np.random.random() < 0.4:  # Probability to change the mutation strategy
            indices = np.random.choice([i for i in range(self.population_size) if i != idx], 3, replace=False)
        else:
            indices = [i for i in range(self.population_size) if i != idx]
            np.random.shuffle(indices)
            a, b, c = indices[:3]

        mutant = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
        adaptive_factor = 0.5 + 0.5 * np.random.rand()
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

    def randomized_newton_update(self, idx, func):
        gradient = np.random.uniform(-1, 1, self.dim)
        adaptive_lr = self.learning_rate / (1 + np.random.uniform(0, 1))  # Change learning rate adaptively
        hessian_approx = np.random.uniform(0.1, 0.5, self.dim)
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

                self.randomized_newton_update(i, func)
                evaluations += 1

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]