import numpy as np

class ES_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.max_iter = budget // self.population_size

    def __call__(self, func):
        def objective_function(x):
            return func(x)

        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        def mutate(x, sigma):
            return x + np.random.normal(0, sigma, size=self.dim)

        population = initialize_population()
        best_solution = population[np.argmin([objective_function(ind) for ind in population]).copy()
        sigma = 0.1

        for _ in range(self.max_iter):
            for i in range(self.population_size):
                new_solution = mutate(population[i], sigma)
                if objective_function(new_solution) < objective_function(population[i]):
                    population[i] = new_solution
                    if objective_function(new_solution) < objective_function(best_solution):
                        best_solution = new_solution
                else:
                    acceptance_probability = np.exp((objective_function(population[i]) - objective_function(new_solution)) / 1)
                    if np.random.rand() < acceptance_probability:
                        population[i] = new_solution

            sigma *= 0.99  # Annealing the standard deviation

        return best_solution