import numpy as np

class GeneticAnnealingAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def mutate(self, x):
        return x + np.random.uniform(-1, 1, self.dim) * self.mutation_rate

    def crossover(self, parent1, parent2):
        mask = np.random.choice([0, 1], self.dim)
        child = parent1 * mask + parent2 * (1 - mask)
        return child

    def annealing(self, x, t):
        return x + np.random.normal(0, t, self.dim)

    def acceptance_probability(self, current, new, t):
        if new < current:
            return 1.0
        return np.exp(- (new - current) / t)

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.pop_size, self.dim))
        current_best = population[np.argmin([func(p) for p in population])]
        t = 1.0

        for _ in range(self.budget - self.pop_size):
            new_population = []
            for _ in range(self.pop_size):
                selected = np.random.choice(self.pop_size, 2, replace=False)
                parent1, parent2 = population[selected[0]], population[selected[1]]

                child = self.crossover(parent1, parent2) if np.random.rand() < self.crossover_rate else parent1
                child = self.mutate(child)
                new_population.append(child)

                new_t = t * 0.99
                new_child = self.annealing(child, new_t)
                if np.random.rand() < self.acceptance_probability(func(child), func(new_child), new_t):
                    new_population[-1] = new_child

            population = np.array(new_population)
            current_best = population[np.argmin([func(p) for p in population])]
            t *= 0.95

        return current_best