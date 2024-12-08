import numpy as np
from scipy.optimize import differential_evolution

class DPHE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def neg_func(x):
            return -func(x)

        bounds = [(self.lower_bound, self.upper_bound) for _ in range(self.dim)]
        res = differential_evolution(func, bounds, x0=np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim), maxiter=self.budget, tol=1e-6)

        if res.success:
            return res.x
        else:
            return None

def mutation(individual, logger):
    if np.random.rand() < 0.45:
        i = np.random.randint(0, len(individual))
        j = np.random.randint(0, len(individual))
        individual[i], individual[j] = individual[j], individual[i]
    return individual

def crossover(individual1, individual2, logger):
    i = np.random.randint(0, len(individual1))
    j = np.random.randint(0, len(individual2))
    child = individual1[:i] + individual2[j:]
    return child

def dphe_algorithm(func, dim, budget):
    dphe = DPHE(budget, dim)
    population = [np.random.uniform(-5.0, 5.0, size=dim) for _ in range(100)]
    logger = {}
    for _ in range(100):
        new_population = []
        for individual in population:
            result = dphe(func, individual)
            if result is not None:
                new_population.append(result)
            else:
                new_population.append(individual)
        population = new_population
        if len(population) > 100:
            population = population[:100]
        np.random.shuffle(population)
    return population

# Example usage:
if __name__ == "__main__":
    # Define a sample black box function
    def func(x):
        return np.sum(x**2)

    # Optimize the function
    dim = 10
    budget = 100
    population = dphe_algorithm(func, dim, budget)
    print("Optimal solution:", population[0])