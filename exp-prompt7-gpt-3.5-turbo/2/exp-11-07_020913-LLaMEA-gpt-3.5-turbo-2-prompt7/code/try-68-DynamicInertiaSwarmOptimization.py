import numpy as np

class DynamicInertiaSwarmOptimization:
    def __init__(self, budget, dim, num_particles=30, alpha=0.5, beta=2.0, gamma=1.0, inertia_max=0.9, inertia_min=0.4):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.inertia_max = inertia_max
        self.inertia_min = inertia_min

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))
        velocity = np.zeros((self.num_particles, self.dim))
        fitness = np.array([func(individual) for individual in population])
        best_global_pos = population[np.argmin(fitness)]
        inertia_weight = self.inertia_max

        for _ in range(self.budget - self.num_particles):
            r1, r2 = np.random.rand(2, self.num_particles, self.dim)
            velocity = inertia_weight * velocity + self.alpha * r1 * (best_global_pos - population) + self.beta * r2 * (population - np.mean(population, axis=0))
            population += velocity
            fitness = np.array([func(individual) for individual in population])
            best_global_pos = population[np.argmin(fitness)]
            inertia_weight = self.inertia_max - ((_ + 1) / (self.budget - self.num_particles)) * (self.inertia_max - self.inertia_min)

        return best_global_pos