import numpy as np

class QuantumGravitationalAlgorithm:
    def __init__(self, budget, dim, num_particles=30, num_iterations=1000, alpha=0.2, beta=0.5, G=6.67430e-11):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.G = G

    def quantum_rotation_gate(self, population):
        rotation_angle = np.random.uniform(0, 2*np.pi, size=(self.num_particles, self.dim))
        population *= np.exp(1j * rotation_angle)
        return population

    def gravitational_search(self, population, fitness_values):
        mass = 1.0 / (fitness_values + 1e-10)
        force = np.zeros((self.num_particles, self.dim))
        for i in range(self.num_particles):
            for j in range(self.num_particles):
                if i != j:
                    r = population[j] - population[i]
                    force[i] += self.G * mass[i] * mass[j] * r / (np.linalg.norm(r) + 1e-10)
        return force

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))
        for _ in range(self.num_iterations):
            population = self.quantum_rotation_gate(population)
            fitness_values = [func(individual) for individual in population]
            force = self.gravitational_search(population, fitness_values)
            population += self.beta * force
            best_individual = population[np.argmin(fitness_values)]
            population = self.alpha * best_individual + np.sqrt(1 - self.alpha**2) * population
        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution