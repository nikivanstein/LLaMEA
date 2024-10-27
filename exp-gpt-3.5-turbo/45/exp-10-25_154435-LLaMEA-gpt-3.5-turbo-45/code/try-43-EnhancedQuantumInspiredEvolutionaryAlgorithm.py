import numpy as np

class EnhancedQuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim, num_particles=30, num_iterations=1000, alpha=0.2, beta=0.5, differential_weight=0.5, crossover_prob=0.7):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.differential_weight = differential_weight
        self.crossover_prob = crossover_prob

    def quantum_rotation_gate(self, population):
        rotation_angle = np.random.uniform(0, 2*np.pi, size=(self.dim, 1))
        population *= np.exp(1j * rotation_angle)
        return population

    def differential_evolution(self, population, best_individual):
        F = np.random.uniform(0, self.differential_weight, size=(self.num_particles, self.dim))
        CR = np.random.uniform(0, self.crossover_prob, size=self.num_particles)
        for i in range(self.num_particles):
            r1, r2, r3 = np.random.choice(self.num_particles, 3, replace=False)
            mutant = population[r1] + F[i] * (population[r2] - population[r3])
            crossover_mask = np.random.rand(self.dim) < CR[i]
            population[i] = np.where(crossover_mask, mutant, population[i])
        return population

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))
        for _ in range(self.num_iterations):
            population = self.quantum_rotation_gate(population)
            population = self.differential_evolution(population, population[np.argmin([func(individual) for individual in population])])
            best_solution = population[np.argmin([func(individual) for individual in population])]
            population = self.alpha * best_solution + np.sqrt(1 - self.alpha**2) * population
        return best_solution