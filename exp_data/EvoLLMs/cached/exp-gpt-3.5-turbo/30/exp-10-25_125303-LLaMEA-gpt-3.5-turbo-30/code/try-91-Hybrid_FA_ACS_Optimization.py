import numpy as np

class Hybrid_FA_ACS_Optimization:
    def __init__(self, budget, dim, population_size=50, alpha=0.1, beta=1.5, gamma=0.3, pa=0.25, step_size=0.1, mutation_prob=0.3):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.pa = pa
        self.step_size = step_size
        self.mutation_prob = mutation_prob

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        def hybrid_fa_acs_step(population, best_individual):
            new_population = []
            for idx, target in enumerate(population):
                new_individual = np.zeros_like(target)
                for i in range(len(target)):
                    new_individual[i] = target[i] + self.step_size * np.random.randn()  # Firefly movement
                    if np.random.rand() < self.pa:
                        cuckoo = target + self.beta * np.random.randn(self.dim)  # Cuckoo search
                        if func(cuckoo) < func(new_individual):
                            new_individual = cuckoo
                    if np.random.rand() < self.mutation_prob:
                        new_individual[i] += np.random.normal(0, 0.1)  # Adaptive mutation
                new_population.append(new_individual)
            return np.array(new_population)

        population = initialize_population()
        best_individual = population[np.argmin([func(ind) for ind in population])]
        remaining_budget = self.budget - self.population_size

        while remaining_budget > 0:
            new_population = hybrid_fa_acs_step(population, best_individual)
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
# optimizer = Hybrid_FA_ACS_Optimization(budget=1000, dim=10)
# result = optimizer(lambda x: np.sum(x**2))  # Optimize the sphere function