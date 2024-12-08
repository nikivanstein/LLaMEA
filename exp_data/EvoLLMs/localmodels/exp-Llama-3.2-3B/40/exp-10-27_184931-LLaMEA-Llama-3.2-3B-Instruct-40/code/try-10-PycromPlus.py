import numpy as np
from scipy.optimize import differential_evolution

class PycromPlus:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_rate = 0.1
        self.tau = 0.9
        self.refine_rate = 0.4

    def __call__(self, func):
        if self.budget == 0:
            return None

        # Initialize the population with random solutions
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        for _ in range(self.budget):
            # Evaluate the fitness of each solution
            fitness = np.array([func(x) for x in population])

            # Select the fittest solutions
            fittest_idx = np.argsort(fitness)[:self.population_size // 2]
            population = population[fittest_idx]

            # Perform differential evolution to generate new solutions
            new_population = differential_evolution(lambda x: func(x), [(-5.0, 5.0) for _ in range(self.dim)], x0=population)

            # Update the population
            population = np.concatenate((population, new_population))

            # Apply mutation to some solutions
            idx = np.random.choice(self.population_size, size=int(self.population_size * self.mutation_rate), replace=False)
            population[idx] = np.random.uniform(-5.0, 5.0, size=(len(idx), self.dim))

        # Apply simulated annealing to the population
        temperature = 1000.0
        for _ in range(self.budget // 2):
            # Select a random solution
            idx = np.random.choice(self.population_size)
            solution = population[idx]

            # Generate a new solution
            new_solution = solution + np.random.normal(0, 1, size=self.dim)

            # Calculate the difference in fitness
            delta_fitness = func(new_solution) - func(solution)

            # Accept the new solution if it's better or with a certain probability
            if delta_fitness > 0 or np.random.rand() < np.exp(-(delta_fitness / temperature)):
                population[idx] = new_solution

            # Decrease the temperature
            temperature *= self.tau

        # Refine the population by changing individual lines
        for _ in range(int(self.budget * self.refine_rate)):
            idx = np.random.choice(self.population_size, size=int(self.population_size * self.refine_rate), replace=False)
            new_individual = population[idx]
            new_individual[np.random.choice(self.dim, size=int(self.dim * self.refine_rate)), :] = np.random.uniform(-5.0, 5.0, size=(len(idx), self.dim))
            population[idx] = new_individual

        # Return the best solution
        return np.min(population, axis=0)

# Test the algorithm
def func(x):
    return np.sum(x**2)

budget = 100
dim = 5
algorithm = PycromPlus(budget, dim)
solution = algorithm(func)
print(solution)