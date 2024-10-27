import numpy as np

class PSO_ES_Hybrid:
    def __init__(self, budget, dim, population_size=30, c1=2.0, c2=2.0, sigma=0.1):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.c1 = c1
        self.c2 = c2
        self.sigma = sigma

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        def pso_es_step(population, velocities, best_individual):
            new_population = []
            for idx, particle in enumerate(population):
                velocity = velocities[idx]
                new_velocity = self.c1 * np.random.rand() * (best_individual - particle) + self.c2 * np.random.rand() * (population[np.random.randint(0, self.population_size)] - particle)
                new_particle = particle + new_velocity + self.sigma * np.random.randn(self.dim)
                new_population.append(new_particle)
            return np.array(new_population)

        population = initialize_population()
        velocities = [np.zeros(self.dim) for _ in range(self.population_size)]
        best_individual = population[np.argmin([func(ind) for ind in population])]
        remaining_budget = self.budget - self.population_size

        while remaining_budget > 0:
            new_population = pso_es_step(population, velocities, best_individual)
            for idx, individual in enumerate(new_population):
                if remaining_budget <= 0:
                    break
                new_fitness = func(individual)
                if new_fitness < func(population[idx]):
                    population[idx] = individual
                    velocities[idx] = individual - population[idx]
                    if new_fitness < func(best_individual):
                        best_individual = individual
                remaining_budget -= 1

        return best_individual

# Example usage:
# optimizer = PSO_ES_Hybrid(budget=1000, dim=10)
# result = optimizer(lambda x: np.sum(x**2))  # Optimize the sphere function