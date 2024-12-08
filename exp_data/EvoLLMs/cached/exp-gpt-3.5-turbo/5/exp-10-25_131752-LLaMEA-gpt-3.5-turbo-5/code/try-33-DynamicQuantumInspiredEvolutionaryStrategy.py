import numpy as np

class DynamicQuantumInspiredEvolutionaryStrategy:
    def __init__(self, budget, dim, mu=5, lambda_=20, sigma=0.1, phase_factor=0.5):
        self.budget = budget
        self.dim = dim
        self.mu = mu
        self.lambda_ = lambda_
        self.sigma = sigma
        self.phase_factor = phase_factor

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, (self.mu, self.dim))

        def apply_quantum_phase(population, phase):
            return np.multiply(population, np.exp(1j * phase))

        def mutate_population(population, phase):
            offspring_population = np.clip(population + self.sigma * np.random.randn(self.lambda_, self.dim), -5.0, 5.0)
            return apply_quantum_phase(offspring_population, phase)

        population = initialize_population()
        for _ in range(self.budget):
            phase = np.random.uniform(0, 2 * np.pi)
            offspring = mutate_population(population, self.phase_factor)
            fitness = np.array([func(individual) for individual in offspring])
            best_index = np.argmin(fitness)
            if fitness[best_index] < func(population[0]):
                population[0] = offspring[best_index]
            self.phase_factor = np.random.uniform(0, 1)  # Dynamic mutation phase
        return population[0]