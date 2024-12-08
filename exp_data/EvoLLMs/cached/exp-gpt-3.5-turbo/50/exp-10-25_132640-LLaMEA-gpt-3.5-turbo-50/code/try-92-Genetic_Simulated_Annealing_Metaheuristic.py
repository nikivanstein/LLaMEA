import numpy as np

class Genetic_Simulated_Annealing_Metaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = 0.1
        self.initial_temperature = 100.0

    def __call__(self, func):
        population = np.random.uniform(low=-5.0, high=5.0, size=(self.budget, self.dim))
        best_solution = population[np.argmin(func(population))]
        temperature = self.initial_temperature

        for _ in range(self.budget):
            mutation_prob = np.random.rand()
            new_solution = best_solution.copy()

            if mutation_prob < self.mutation_rate:
                mutation_idx = np.random.randint(self.dim)
                new_solution[mutation_idx] += np.random.uniform(-0.5, 0.5)

            cost_diff = func(new_solution) - func(best_solution)
            if cost_diff < 0 or np.random.rand() < np.exp(-cost_diff / temperature):
                best_solution = new_solution

            temperature *= 0.95  # Cooling schedule

        return best_solution