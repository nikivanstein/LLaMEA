import numpy as np
import random

class SwarmOptimizationInspiredAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.crossover_probability = 0.8
        self.mutation_probability = 0.1
        self.swarm_particles = self.initialize_swarm_particles()
        self.best_particles = self.initialize_best_particles()

    def initialize_swarm_particles(self):
        return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def initialize_best_particles(self):
        return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def evaluate(self, func):
        fitnesses = func(self.swarm_particles)
        self.swarm_particles = self.select_swarm_particles(fitnesses)
        self.swarm_particles = self.crossover(self.swarm_particles)
        self.swarm_particles = self.mutate(self.swarm_particles)
        self.best_particles = self.update_best_particles(fitnesses)

    def select_swarm_particles(self, fitnesses):
        fitnesses = np.array(fitnesses)
        swarm_particles = np.array([self.swarm_particles[np.argsort(fitnesses)[:int(self.population_size/2)]]])
        return swarm_particles

    def crossover(self, population):
        offspring = np.zeros((self.population_size, self.dim))
        for i in range(self.population_size):
            if random.random() < self.crossover_probability:
                parent1 = random.choice(population)
                parent2 = random.choice(population)
                offspring[i] = (parent1 + parent2) / 2
        return offspring

    def mutate(self, population):
        mutated_population = np.copy(population)
        for i in range(self.population_size):
            if random.random() < self.mutation_probability:
                mutated_population[i] += np.random.uniform(-1.0, 1.0, self.dim)
        return mutated_population

    def update_best_particles(self, fitnesses):
        fitnesses = np.array(fitnesses)
        best_particles = np.array([self.swarm_particles[np.argsort(fitnesses)[-1:]]])
        return best_particles

    def __call__(self, func):
        for _ in range(self.budget):
            self.evaluate(func)
        return np.min(self.best_particles, axis=0)

# Example usage
def func(x):
    return np.sum(x**2)

ssoa = SwarmOptimizationInspiredAlgorithm(budget=100, dim=10)
optimal_solution = ssoa(func)
print(optimal_solution)