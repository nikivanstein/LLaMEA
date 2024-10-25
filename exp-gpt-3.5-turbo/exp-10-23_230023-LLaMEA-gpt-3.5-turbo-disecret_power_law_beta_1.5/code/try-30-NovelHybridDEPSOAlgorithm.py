import numpy as np

class NovelHybridDEPSOAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.mutation_prob = 0.5
        self.novel_mutation_scale = 0.2  # Updated mutation scale parameter
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.c1 = 1.49445  # PSO cognitive component
        self.c2 = 1.49445  # PSO social component
        self.w = 0.729  # PSO inertia weight

    def mutate(self, x, pop):
        idxs = np.random.choice(len(pop), 2, replace=False)
        a, b = pop[idxs[0]], pop[idxs[1]]
        return x + np.random.uniform(-self.novel_mutation_scale, self.novel_mutation_scale) * (a - b)

    def update_velocity(self, x, pbest, gbest, velocity):
        return self.w * velocity + self.c1 * np.random.rand() * (pbest - x) + self.c2 * np.random.rand() * (gbest - x)

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.pop_size, self.dim))
        velocity = np.zeros((self.pop_size, self.dim))
        pbest = population.copy()
        gbest = population[np.argmin([func(individual) for individual in population])]

        for _ in range(self.budget - self.pop_size):
            for i in range(self.pop_size):
                velocity[i] = self.update_velocity(population[i], pbest[i], gbest, velocity[i])
                new_individual = population[i] + velocity[i]
                new_individual = self.mutate(new_individual, population)
                new_fitness = func(new_individual)

                if new_fitness < func(population[i]):
                    population[i] = new_individual
                    pbest[i] = new_individual

                if new_fitness < func(gbest):
                    gbest = new_individual

        return gbest