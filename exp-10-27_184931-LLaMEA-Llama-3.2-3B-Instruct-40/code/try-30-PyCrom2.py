import numpy as np
from scipy.optimize import differential_evolution

class PyCrom2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_rate = 0.1
        self.tau = 0.9
        self.probability = 0.4

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

            # Apply mutation to some solutions with probability
            idx = np.random.choice(self.population_size, size=int(self.population_size * self.mutation_rate), replace=False)
            for i in idx:
                if np.random.rand() < self.probability:
                    population[i] = np.random.uniform(-5.0, 5.0, size=self.dim)

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

        # Return the best solution
        return np.min(population, axis=0)

# Usage
def bbb(f, bounds, x0=None):
    if x0 is None:
        x0 = np.array([0.5] * len(f))
    bounds = np.array(bounds)
    if len(bounds)!= len(x0):
        raise ValueError("Number of variables and bounds do not match")
    return f(x0)

def bbb_test_suite():
    functions = [
        lambda x: x[0]**2 + x[1]**2,
        lambda x: x[0]**2 + x[1]**3,
        lambda x: x[0]**4 + x[1]**4,
        lambda x: x[0]**4 + x[1]**2,
        lambda x: x[0]**2 + x[1]**3 + x[2]**2 + x[3]**2,
        lambda x: x[0]**2 + x[1]**3 + x[2]**2 + x[3]**2 + x[4]**2,
        lambda x: x[0]**4 + x[1]**4 + x[2]**4 + x[3]**4 + x[4]**4,
        lambda x: x[0]**4 + x[1]**4 + x[2]**4 + x[3]**4 + x[4]**4 + x[5]**4,
        lambda x: x[0]**2 + x[1]**3 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2,
        lambda x: x[0]**4 + x[1]**4 + x[2]**4 + x[3]**4 + x[4]**4 + x[5]**4 + x[6]**4,
        lambda x: x[0]**2 + x[1]**3 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2,
        lambda x: x[0]**4 + x[1]**4 + x[2]**4 + x[3]**4 + x[4]**4 + x[5]**4 + x[6]**4 + x[7]**4,
        lambda x: x[0]**2 + x[1]**3 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2,
        lambda x: x[0]**4 + x[1]**4 + x[2]**4 + x[3]**4 + x[4]**4 + x[5]**4 + x[6]**4 + x[7]**4 + x[8]**4,
        lambda x: x[0]**2 + x[1]**3 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2,
        lambda x: x[0]**4 + x[1]**4 + x[2]**4 + x[3]**4 + x[4]**4 + x[5]**4 + x[6]**4 + x[7]**4 + x[8]**4 + x[9]**4,
        lambda x: x[0]**2 + x[1]**3 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2,
        lambda x: x[0]**4 + x[1]**4 + x[2]**4 + x[3]**4 + x[4]**4 + x[5]**4 + x[6]**4 + x[7]**4 + x[8]**4 + x[9]**4 + x[10]**4,
        lambda x: x[0]**2 + x[1]**3 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2,
        lambda x: x[0]**4 + x[1]**4 + x[2]**4 + x[3]**4 + x[4]**4 + x[5]**4 + x[6]**4 + x[7]**4 + x[8]**4 + x[9]**4 + x[10]**4 + x[11]**4,
        lambda x: x[0]**2 + x[1]**3 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2,
        lambda x: x[0]**4 + x[1]**4 + x[2]**4 + x[3]**4 + x[4]**4 + x[5]**4 + x[6]**4 + x[7]**4 + x[8]**4 + x[9]**4 + x[10]**4 + x[11]**4 + x[12]**4,
    ]

    for f in functions:
        print(f"Function {functions.index(f)+1}: {f.__name__}")
        print("Optimization results:")
        print("Best solution: ", bbb(f, [(-5.0, 5.0) for _ in range(len(f._func_params))]))
        print()

def evaluate_bbb():
    for i in range(24):
        bbb_test_suite()

# Usage
evaluate_bbb()