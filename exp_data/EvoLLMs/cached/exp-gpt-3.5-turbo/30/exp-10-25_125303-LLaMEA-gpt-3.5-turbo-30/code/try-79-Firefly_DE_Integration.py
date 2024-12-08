import numpy as np

class Firefly_DE_Integration:
    def __init__(self, budget, dim, population_size=30, f=0.5, cr=0.9, alpha=0.5, beta_min=0.2):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f = f
        self.cr = cr
        self.alpha = alpha
        self.beta_min = beta_min

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        def firefly_de_step(population, best_individual):
            new_population = []
            for idx, target in enumerate(population):
                attractiveness = np.zeros(self.population_size)
                for j, other in enumerate(population):
                    distance = np.linalg.norm(other - target)
                    attractiveness[j] = self.beta_min + (1.0 - self.beta_min) * np.exp(-self.alpha * distance ** 2)

                best_index = np.argmin([func(ind) for ind in population])
                best_individual = population[best_index]

                for i in range(self.dim):
                    if np.random.rand() < attractiveness[idx]:
                        mutant = target + self.f * (best_individual - target) + self.cr * (population[np.random.randint(0, self.population_size)] - population[np.random.randint(0, self.population_size)])
                        target[i] = mutant[i]

                new_population.append(target)
            return np.array(new_population)

        population = initialize_population()
        best_individual = population[np.argmin([func(ind) for ind in population])]
        remaining_budget = self.budget - self.population_size

        while remaining_budget > 0:
            new_population = firefly_de_step(population, best_individual)
            for idx, individual in enumerate(new_population):
                if remaining_budget <= 0:
                    break
                new_fitness = func(individual)
                if new_fitness < func(population[idx]):
                    population[idx] = individual
                    if new_fitness < func(best_individual):
                        best_individual = individual
                remaining_budget -= 1

        return best_individual

# Example usage:
# optimizer = Firefly_DE_Integration(budget=1000, dim=10)
# result = optimizer(lambda x: np.sum(x**2))  # Optimize the sphere function