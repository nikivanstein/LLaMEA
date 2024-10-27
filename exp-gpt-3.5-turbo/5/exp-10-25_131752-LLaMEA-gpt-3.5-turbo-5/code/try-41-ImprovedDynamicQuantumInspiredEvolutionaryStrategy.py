import numpy as np

class ImprovedDynamicQuantumInspiredEvolutionaryStrategy:
    def __init__(self, budget, dim, mu=5, lambda_=20, sigma=0.1, phase_factor=0.5, f=0.5, cr=0.9):
        self.budget = budget
        self.dim = dim
        self.mu = mu
        self.lambda_ = lambda_
        self.sigma = sigma
        self.phase_factor = phase_factor
        self.f = f
        self.cr = cr

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, (self.mu, self.dim))

        def apply_quantum_phase(population):
            phase = np.random.uniform(0, 2 * np.pi)
            return np.multiply(population, np.exp(1j * self.phase_factor * phase))

        def mutate_population(population):
            offspring_population = np.zeros((self.lambda_, self.dim))
            for i in range(self.lambda_):
                idxs = np.random.choice(range(self.mu), 3, replace=False)
                trial_vector = population[idxs[0]] + self.f * (population[idxs[1]] - population[idxs[2]])
                crossover = np.random.rand(self.dim) < self.cr
                offspring_population[i] = np.where(crossover, trial_vector, population[i])
                offspring_population[i] = np.clip(offspring_population[i], -5.0, 5.0)
            return apply_quantum_phase(offspring_population)

        population = initialize_population()
        for _ in range(self.budget):
            offspring = mutate_population(population)
            fitness = np.array([func(individual) for individual in offspring])
            best_index = np.argmin(fitness)
            if fitness[best_index] < func(population[0]):
                population[0] = offspring[best_index]
            self.phase_factor = np.random.uniform(0, 1)  # Dynamic mutation phase
        return population[0]