import numpy as np

class HybridQuantumEvolutionaryAlgorithm:
    def __init__(self, budget, dim, num_particles=30, num_iterations=1000, alpha=0.2, beta=0.5, cr=0.7, f=0.5):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.cr = cr
        self.f = f

    def differential_evolution(self, population, fitness_func):
        mutant_population = np.zeros_like(population)
        for i in range(self.num_particles):
            idxs = [idx for idx in range(self.num_particles) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = population[i] + self.f * (a - b)
            j_rand = np.random.randint(self.dim)
            for j in range(self.dim):
                if np.random.uniform() < self.cr or j == j_rand:
                    mutant[j] = a[j] + self.f * (b[j] - c[j])
            mutant_population[i] = mutant if fitness_func(mutant) < fitness_func(population[i]) else population[i]
        return mutant_population

    def quantum_rotation_gate(self, population):
        rotation_angle = np.random.uniform(0, 2*np.pi, size=self.dim)
        population *= np.exp(1j * rotation_angle)
        return population

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))
        for _ in range(self.num_iterations):
            population = self.quantum_rotation_gate(population)
            population = self.differential_evolution(population, func)
            best_individual = population[np.argmin([func(individual) for individual in population])]
            population = self.alpha * best_individual + np.sqrt(1-self.alpha**2) * population
        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution