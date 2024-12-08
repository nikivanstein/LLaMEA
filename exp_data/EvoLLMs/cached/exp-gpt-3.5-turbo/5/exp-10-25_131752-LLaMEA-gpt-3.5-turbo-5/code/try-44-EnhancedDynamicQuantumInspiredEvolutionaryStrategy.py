import numpy as np

class EnhancedDynamicQuantumInspiredEvolutionaryStrategy:
    def __init__(self, budget, dim, mu=5, lambda_=20, sigma=0.1, phase_factor=0.5, elitism_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.mu = mu
        self.lambda_ = lambda_
        self.sigma = sigma
        self.phase_factor = phase_factor
        self.elitism_rate = elitism_rate

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, (self.mu, self.dim))

        def apply_quantum_phase(population):
            phase = np.random.uniform(0, 2 * np.pi)
            return np.multiply(population, np.exp(1j * self.phase_factor * phase))

        def mutate_population(population):
            offspring_population = np.zeros((self.lambda_, self.dim))
            for i in range(self.lambda_):
                parent = population[np.random.randint(self.mu)]
                offspring_population[i] = parent + self.sigma * np.random.randn(self.dim)
                offspring_population[i] = np.clip(offspring_population[i], -5.0, 5.0)
            return apply_quantum_phase(offspring_population)

        population = initialize_population()
        for _ in range(self.budget):
            offspring = mutate_population(population)
            fitness = np.array([func(individual) for individual in offspring])
            best_index = np.argmin(fitness)
            if fitness[best_index] < func(population[0]):
                population[0] = offspring[best_index]
            # Introduce elitism
            num_elites = int(self.mu * self.elitism_rate)
            elite_indices = np.argsort(fitness)[:num_elites]
            population[1:num_elites+1] = offspring[elite_indices]
            self.phase_factor = np.random.uniform(0, 1)  # Dynamic mutation phase
        return population[0]