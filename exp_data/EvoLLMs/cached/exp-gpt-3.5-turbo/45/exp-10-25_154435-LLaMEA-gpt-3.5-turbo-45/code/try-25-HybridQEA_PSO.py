import numpy as np

class HybridQEA_PSO:
    def __init__(self, budget, dim, num_particles=30, num_iterations=1000, alpha=0.2, beta=0.5, inertia_weight=0.5, cognitive_param=0.5, social_param=0.5):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.inertia_weight = inertia_weight
        self.cognitive_param = cognitive_param
        self.social_param = social_param

    def quantum_rotation_gate(self, population):
        rotation_angle = np.random.uniform(0, 2*np.pi, size=self.dim)
        population *= np.exp(1j * rotation_angle)
        return population

    def particle_swarm_optimization(self, population, func):
        best_position = population.copy()
        best_fitness = [func(individual) for individual in population]
        for _ in range(self.num_iterations):
            for i in range(self.num_particles):
                cognitive = self.cognitive_param * np.random.rand(self.dim) * (best_position[i] - population[i])
                social = self.social_param * np.random.rand(self.dim) * (best_position[np.argmin(best_fitness)] - population[i])
                velocity = self.inertia_weight * population[i] + cognitive + social
                population[i] += velocity
                if func(population[i]) < best_fitness[i]:
                    best_position[i] = population[i]
                    best_fitness[i] = func(population[i])
        return best_position

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))
        for _ in range(self.num_iterations):
            population = self.quantum_rotation_gate(population)
            population = self.particle_swarm_optimization(population, func)
            best_individual = population[np.argmin([func(individual) for individual in population])]
            population = self.alpha * best_individual + np.sqrt(1 - self.alpha**2) * population
        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution