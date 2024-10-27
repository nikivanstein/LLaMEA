import numpy as np

class EnhancedQuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim, num_particles=30, num_iterations=1000, alpha=0.2, beta=0.5, mutation_prob=0.1, mutation_strength=0.1):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.mutation_prob = mutation_prob
        self.mutation_strength = mutation_strength

    def quantum_rotation_gate(self, population):
        rotation_angle = np.random.uniform(0, 2*np.pi, size=(self.dim, self.num_particles))
        population *= np.exp(1j * rotation_angle)
        return population

    def mutation(self, population):
        mutated_population = population.copy()
        for i in range(self.num_particles):
            if np.random.rand() < self.mutation_prob:
                mutation_vector = np.random.uniform(-self.mutation_strength, self.mutation_strength, size=self.dim)
                mutated_population[i] += mutation_vector
        return mutated_population

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))
        for _ in range(self.num_iterations):
            population = self.quantum_rotation_gate(population)
            population = self.mutation(population)
            fitness_values = [func(individual) for individual in population]
            top_indices = np.argsort(fitness_values)[:self.num_particles//2]
            population = population[top_indices]
            best_individual = population[np.argmin(fitness_values)]
            population = self.alpha * best_individual + np.sqrt(1-self.alpha**2) * population
        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution