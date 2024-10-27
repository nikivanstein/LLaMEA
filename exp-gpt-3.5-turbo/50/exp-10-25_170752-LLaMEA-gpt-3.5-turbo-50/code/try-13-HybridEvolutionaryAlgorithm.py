import numpy as np

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mu = 10
        self.lambda_ = 50
        self.sigma_init = 0.1
        self.sigma_min = 1e-6
        self.inertia_weight = 0.5
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.particles = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.population_size, self.dim))
        self.sigmas = np.full(self.population_size, self.sigma_init)
        self.best_solution = None
        self.best_fitness = float('inf')

    def adapt_sigma(self):
        for i in range(self.population_size):
            self.sigmas[i] = max(self.sigmas[i] * np.exp(0.1 * np.random.normal(0, 1)), self.sigma_min)

    def PSO_update(self, func):
        for i in range(self.population_size):
            cognitive_component = self.cognitive_weight * np.random.rand(self.dim) * (self.best_solution - self.particles[i])
            social_component = self.social_weight * np.random.rand(self.dim) * (self.best_solution - self.particles[i])
            self.velocities[i] = self.inertia_weight * self.velocities[i] + cognitive_component + social_component
            self.particles[i] = np.clip(self.particles[i] + self.velocities[i], -5.0, 5.0)
    
    def ES_update(self, func):
        offspring = np.random.normal(self.particles, self.sigmas)
        fitness_offspring = np.array([func(offspring[i]) for i in range(self.population_size)])
        best_offspring = offspring[np.argmin(fitness_offspring)]
        if func(best_offspring) < func(self.best_solution):
            self.best_solution = best_offspring
        return offspring, fitness_offspring

    def __call__(self, func):
        for _ in range(self.budget):
            self.adapt_sigma()
            offspring, _ = self.ES_update(func)
            self.PSO_update(func)
        return self.best_solution