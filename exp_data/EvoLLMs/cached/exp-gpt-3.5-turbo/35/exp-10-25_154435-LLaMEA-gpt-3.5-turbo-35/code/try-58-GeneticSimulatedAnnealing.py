import numpy as np

class GeneticSimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.max_iters = budget // self.pop_size
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        def initialize_population(dim, pop_size, lb, ub):
            return np.random.uniform(lb, ub, (pop_size, dim))

        def mutate(individual, lb, ub, T):
            new_individual = individual + np.random.normal(0, T, len(individual))
            return np.clip(new_individual, lb, ub)

        def accept_move(new_fitness, current_fitness, T):
            if new_fitness < current_fitness:
                return True
            else:
                acceptance_prob = np.exp((current_fitness - new_fitness) / T)
                return np.random.rand() < acceptance_prob

        population = initialize_population(self.dim, self.pop_size, self.lb, self.ub)
        T = 1.0
        for _ in range(self.max_iters):
            for i in range(self.pop_size):
                current_fitness = func(population[i])
                new_individual = mutate(population[i], self.lb, self.ub, T)
                new_fitness = func(new_individual)
                if accept_move(new_fitness, current_fitness, T):
                    population[i] = new_individual
            T *= 0.95  # Cooling schedule
        best_individual = min(population, key=lambda x: func(x))
        return best_individual