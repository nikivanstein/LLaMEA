import numpy as np

class EnhancedQuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim, num_particles=30, num_iterations=1000, alpha=0.2, beta=0.5, local_search_prob=0.1):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.local_search_prob = local_search_prob

    def quantum_rotation_gate(self, population):
        rotation_angle = np.random.uniform(0, 2*np.pi, size=(self.dim, 1))
        population *= np.exp(1j * rotation_angle)
        return population

    def local_search(self, individual, func):
        candidate = individual.copy()
        for _ in range(5):
            perturbation = np.random.uniform(-0.1, 0.1, size=self.dim)
            candidate += perturbation
            if func(candidate) < func(individual):
                individual = candidate.copy()
        return individual

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))
        for _ in range(self.num_iterations):
            population = self.quantum_rotation_gate(population)
            fitness_values = [func(individual) for individual in population]
            top_indices = np.argsort(fitness_values)[:self.num_particles//2]
            population = population[top_indices]
            if np.random.rand() < self.local_search_prob:
                population = np.array([self.local_search(individual, func) for individual in population])
            best_individual = population[np.argmin(fitness_values)]
            population = self.alpha * best_individual + np.sqrt(1-self.alpha**2) * population
        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution