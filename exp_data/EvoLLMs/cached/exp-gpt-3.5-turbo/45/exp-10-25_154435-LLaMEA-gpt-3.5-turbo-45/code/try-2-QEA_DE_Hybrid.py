import numpy as np

class QEA_DE_Hybrid:
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

    def differential_evolution(self, population, func):
        for i in range(self.num_particles):
            indices = [idx for idx in range(self.num_particles) if idx != i]
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            mutant = a + self.differential_weight * (b - c)
            crossover_mask = np.random.rand(self.dim) < self.crossover_probability
            trial = np.where(crossover_mask, mutant, population[i])
            if func(trial) < func(population[i]):
                population[i] = trial
        return population

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))
        for _ in range(self.num_iterations):
            population = self.quantum_rotation_gate(population)
            population = self.differential_evolution(population, func)
            fitness_values = [func(individual) for individual in population]
            top_indices = np.argsort(fitness_values)[:self.num_particles//2]
            population = population[top_indices]
            best_individual = population[np.argmin(fitness_values)]
            population = self.alpha * best_individual + np.sqrt(1-self.alpha**2) * population
        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution