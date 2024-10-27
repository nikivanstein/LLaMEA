import numpy as np

class DifferentialQuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim, num_particles=30, num_iterations=1000, alpha=0.2, beta=0.5, differential_weight=0.5, crossover_probability=0.9):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.differential_weight = differential_weight
        self.crossover_probability = crossover_probability

    def quantum_rotation_gate(self, population):
        rotation_angle = np.random.uniform(0, 2*np.pi, size=(self.dim, self.num_particles))
        population = np.transpose(population) * np.exp(1j * rotation_angle)
        return np.transpose(population)

    def differential_evolution(self, population, fitness_values):
        mutated_population = population + self.differential_weight * (population[np.random.choice(range(self.num_particles), size=(self.num_particles, self.dim))] - population)
        crossover_mask = np.random.rand(self.num_particles, self.dim) < self.crossover_probability
        trial_population = np.where(crossover_mask, mutated_population, population)
        trial_fitness_values = [func(individual) for individual in trial_population]
        return trial_population, trial_fitness_values

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))
        for _ in range(self.num_iterations):
            population = self.quantum_rotation_gate(population)
            fitness_values = [func(individual) for individual in population]
            top_indices = np.argsort(fitness_values)[:self.num_particles//2]
            best_individual = population[np.argmin(fitness_values)]
            updated_population = self.alpha * best_individual + np.sqrt(1-self.alpha**2) * population
            updated_population, updated_fitness_values = self.differential_evolution(updated_population, fitness_values)
            population = updated_population if min(updated_fitness_values) < min(fitness_values) else population
        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution