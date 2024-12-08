import numpy as np

class ImprovedQuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim, num_particles=30, num_iterations=1000, alpha=0.2, beta=0.5):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta

    def quantum_rotation_gate(self, population):
        rotation_angle = np.random.uniform(0, 2*np.pi, size=(self.num_particles, self.dim))
        population *= np.exp(1j * rotation_angle)
        return population

    def selection(self, population, fitness_values):
        top_indices = np.argsort(fitness_values)[:self.num_particles//2]
        return population[top_indices]

    def update_population(self, population, best_individual):
        return self.alpha * best_individual + np.sqrt(1 - self.alpha**2) * population

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))
        for _ in range(self.num_iterations):
            population = self.quantum_rotation_gate(population)
            fitness_values = [func(individual) for individual in population]
            population = self.selection(population, fitness_values)
            best_individual = population[np.argmin(fitness_values)]
            population = self.update_population(population, best_individual)
        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution