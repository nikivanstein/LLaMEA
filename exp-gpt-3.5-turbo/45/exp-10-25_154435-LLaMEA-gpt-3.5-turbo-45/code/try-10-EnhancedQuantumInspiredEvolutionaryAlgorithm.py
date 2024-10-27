import numpy as np

class EnhancedQuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim, num_particles=30, num_iterations=1000, alpha=0.2, beta=0.5, mutation_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.mutation_rate = mutation_rate

    def quantum_rotation_gate(self, population):
        rotation_angle = np.random.uniform(0, 2*np.pi, size=(self.num_particles, self.dim))
        population *= np.exp(1j * rotation_angle)
        return population

    def mutation_operator(self, individual):
        mutation_indices = np.random.choice([True, False], size=self.dim, p=[self.mutation_rate, 1-self.mutation_rate])
        individual[mutation_indices] += np.random.normal(0, 1, np.sum(mutation_indices))
        return individual

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))
        for _ in range(self.num_iterations):
            population = self.quantum_rotation_gate(population)
            fitness_values = [func(individual) for individual in population]
            top_indices = np.argsort(fitness_values)[:self.num_particles//2]
            population = population[top_indices]
            best_individual = population[np.argmin(fitness_values)]
            mutated_population = np.array([self.mutation_operator(individual) for individual in population])
            population = self.alpha * best_individual + np.sqrt(1-self.alpha**2) * mutated_population
        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution