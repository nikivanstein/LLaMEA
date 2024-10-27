import numpy as np

class PSO_ADE:
    def __init__(self, budget, dim, population_size=30, c1=1.494, c2=1.494, f=0.5, cr=0.9, mutation_prob=0.3):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.c1 = c1
        self.c2 = c2
        self.f = f
        self.cr = cr
        self.mutation_prob = mutation_prob

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        def pso_ade_step(population):
            new_population = []
            for idx, target in enumerate(population):
                pbest = population[np.argmin([func(ind) for ind in population])]
                gbest = population[np.argmin([func(ind) for ind in population])]
                velocity = np.random.uniform(-1, 1, size=self.dim)
                velocity = self.c1 * np.random.rand() * (pbest - target) + self.c2 * np.random.rand() * (gbest - target)
                trial = target + velocity

                for i in range(len(target)):
                    if np.random.rand() < self.cr:
                        trial[i] = target[i] + self.f * (population[np.random.randint(0, self.population_size)][i] - target[i])
                    if np.random.rand() < self.mutation_prob:
                        trial[i] += np.random.normal(0, 0.1)  # Adaptive mutation
                new_population.append(trial)
            return np.array(new_population)

        population = initialize_population()
        remaining_budget = self.budget - self.population_size

        while remaining_budget > 0:
            new_population = pso_ade_step(population)
            for idx, individual in enumerate(new_population):
                if remaining_budget <= 0:
                    break
                new_fitness = func(individual)
                if new_fitness < func(population[idx]):
                    population[idx] = individual
                remaining_budget -= 1

        return population[np.argmin([func(ind) for ind in population])]
# Example usage:
# optimizer = PSO_ADE(budget=1000, dim=10)
# result = optimizer(lambda x: np.sum(x**2))  # Optimize the sphere function