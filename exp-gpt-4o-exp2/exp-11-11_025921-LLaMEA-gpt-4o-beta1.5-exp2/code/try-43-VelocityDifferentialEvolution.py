import numpy as np

class VelocityDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.initial_population_size = min(100, 10 * dim)
        self.population_size = self.initial_population_size
        self.F = 0.7  # Reduced Differential weight
        self.CR = 0.8  # Reduced base crossover probability
        self.convergence_threshold = 0.01
        self.velocity_scale = 0.5  # New: Velocity scale

    def mutation(self, population, velocities):
        idxs = np.random.choice(self.population_size, 3, replace=False)
        diversity = np.linalg.norm(population[idxs[1]] - population[idxs[2]])
        adaptive_F = self.F * (1 + diversity / (self.bounds[1] - self.bounds[0]))
        new_velocity = velocities[idxs[0]] + adaptive_F * (population[idxs[1]] - population[idxs[2]])
        return new_velocity

    def crossover(self, target, mutant, fitness_improvement):
        adaptive_CR = self.CR * (1 + fitness_improvement)
        adaptive_CR = min(1.0, max(0.1, adaptive_CR))
        crossover_mask = np.random.rand(self.dim) < adaptive_CR
        return np.where(crossover_mask, mutant, target)

    def select(self, trial, target, func):
        trial_fitness = func(trial)
        target_fitness = func(target)
        return (trial, trial_fitness) if trial_fitness < target_fitness else (target, target_fitness)

    def manage_diversity(self, population):
        centroid = np.mean(population, axis=0)
        for i in range(self.population_size):
            if np.linalg.norm(population[i] - centroid) < 0.5 * (self.bounds[1] - self.bounds[0]):
                population[i] = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
    
    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = self.population_size

        while evaluations < self.budget:
            self.manage_diversity(population)
            for i in range(self.population_size):
                velocities[i] = self.mutation(population, velocities)
                velocities[i] = np.clip(velocities[i], -self.velocity_scale, self.velocity_scale)
                mutant = population[i] + velocities[i]
                mutant = np.clip(mutant, *self.bounds)
                fitness_improvement = (np.min(fitness) - fitness[i]) / (np.max(fitness) - np.min(fitness) + 1e-10)
                trial = self.crossover(population[i], mutant, fitness_improvement)
                trial = np.clip(trial, *self.bounds)

                new_candidate, new_fitness = self.select(trial, population[i], func)
                evaluations += 1

                if new_fitness < fitness[i]:
                    population[i] = new_candidate
                    fitness[i] = new_fitness

                if evaluations >= self.budget:
                    break

        return population[np.argmin(fitness)]