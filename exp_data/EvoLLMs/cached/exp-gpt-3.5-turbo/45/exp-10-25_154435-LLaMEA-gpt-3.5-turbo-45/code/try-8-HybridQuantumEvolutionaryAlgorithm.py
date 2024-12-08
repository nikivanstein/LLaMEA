import numpy as np

class HybridQuantumEvolutionaryAlgorithm:
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
        rotation_angle = np.random.uniform(0, 2*np.pi, size=(self.num_particles, self.dim))
        population *= np.exp(1j * rotation_angle)
        return population

    def differential_evolution(self, population, best_individual):
        mutant_population = best_individual + self.differential_weight * (population - population[np.random.randint(self.num_particles)])
        crossover_mask = np.random.uniform(0, 1, size=(self.num_particles, self.dim)) < self.crossover_probability
        population[crossover_mask] = mutant_population[crossover_mask]
        return population

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))
        for _ in range(self.num_iterations):
            population = self.quantum_rotation_gate(population)
            fitness_values = [func(individual) for individual in population]
            top_indices = np.argsort(fitness_values)[:self.num_particles//2]
            population = population[top_indices]
            best_individual = population[np.argmin(fitness_values)]
            population = self.alpha * best_individual + np.sqrt(1-self.alpha**2) * population
            population = self.differential_evolution(population, best_individual)
        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution