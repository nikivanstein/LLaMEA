import numpy as np

class EDHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.hmcr = 0.9
        self.par = 0.6
        self.alpha = 0.9
        self.mutation_rate = 0.5

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(low=-5.0, high=5.0, size=(self.pop_size, self.dim))

        def mutate(population, best):
            mutant = population[np.random.choice(range(self.pop_size))]
            trial = best + self.alpha * (mutant - population[np.random.choice(range(self.pop_size))])
            for i in range(self.dim):
                if np.random.rand() < self.mutation_rate:
                    trial[i] = np.random.uniform(low=-5.0, high=5.0)
            return trial

        def evaluate_population(population):
            return np.array([func(individual) for individual in population])

        population = initialize_population()
        costs = evaluate_population(population)
        best_idx = np.argmin(costs)
        best_solution = population[best_idx]
        
        for _ in range(self.budget - self.pop_size):
            new_solution = mutate(population, best_solution)
            new_cost = func(new_solution)
            if new_cost < costs[best_idx]:
                population[best_idx] = new_solution
                costs[best_idx] = new_cost
                best_solution = new_solution
            if np.random.rand() < self.hmcr:
                harmony = np.array([population[np.random.choice(range(self.pop_size)), i] if np.random.rand() < self.par else best_solution[i] for i in range(self.dim)])
                harmony_cost = func(harmony)
                if harmony_cost < costs[best_idx]:
                    population[best_idx] = harmony
                    costs[best_idx] = harmony_cost
                    best_solution = harmony
        
        return best_solution