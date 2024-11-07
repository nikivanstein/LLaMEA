import numpy as np
from scipy.optimize import differential_evolution
from deap import base, creator, tools, algorithms

class CrowdSourcedMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.array([-5.0, 5.0])  # Search space between -5.0 and 5.0
        self.mean = np.random.uniform(self.search_space[0], self.search_space[1], size=self.dim)  # Initialize mean
        self.covariance = np.eye(self.dim) * 1.0  # Initialize covariance matrix

    def __call__(self, func):
        for _ in range(self.budget):
            # Perform evolution strategy to update mean
            new_mean = self.mean + np.random.normal(0, 1.0, size=self.dim)
            new_mean = np.clip(new_mean, self.search_space[0], self.search_space[1])  # Clip values to search space

            # Perform genetic drift to update covariance
            new_covariance = self.covariance + np.random.normal(0, 0.1, size=(self.dim, self.dim))
            new_covariance = np.clip(new_covariance, 0, 1.0)  # Clip values to avoid negative covariance matrix

            # Evaluate function at new mean
            f_new = func(new_mean)

            # Update mean and covariance
            self.mean = new_mean
            self.covariance = new_covariance

            # Print current best solution
            print(f"Current best solution: x = {self.mean}, f(x) = {f_new}")

        # Perform hybrid optimization using Differential Evolution and Genetic Algorithm
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
        toolbox = base.Toolbox()
        toolbox.register("attr_float", np.random.uniform, self.search_space[0], self.search_space[1])
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, self.dim)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", func)
        toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=self.search_space[0], up=self.search_space[1])
        toolbox.register("mutate", tools.mutGaussian, mu=self.mean, sigma=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=self.dim)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)

        for _ in range(10):  # Perform 10 generations
            offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.1)
            pop = toolbox.select(offspring, k=self.dim)
            hof.update(pop)
            record = stats.compile(pop)
            print(record)

        # Perform final optimization using Differential Evolution
        bounds = [(self.search_space[0], self.search_space[1]) for _ in range(self.dim)]
        res = differential_evolution(func, bounds, x0=self.mean, seed=42)
        print(f"Final best solution: x = {res.x}, f(x) = {res.fun}")

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

crowd_sourced = CrowdSourcedMetaheuristic(budget=100, dim=2)
crowd_sourced(func)
