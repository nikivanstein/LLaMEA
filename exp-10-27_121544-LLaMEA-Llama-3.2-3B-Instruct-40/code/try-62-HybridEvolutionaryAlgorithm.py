import numpy as np
from scipy.optimize import differential_evolution

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        # Define the probability distribution for refining the strategy
        strategy_distribution = np.random.choice(['sbx', 'rand1'], p=[0.3, 0.7])

        # Define the probability distribution for scaling
        scaling_distribution = np.random.choice(['uniform', 'log-uniform'], p=[0.4, 0.6])

        # Initialize the population
        population = [np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim) for _ in range(10)]

        # Evaluate the fitness of each individual in the population
        for individual in population:
            exec(f"individual = {func}({individual})")

        # Refine the strategy for each individual in the population
        refined_population = []
        for individual in population:
            strategy = np.random.choice(['sbx', 'rand1'])
            scaling = np.random.choice(['uniform', 'log-uniform'])
            exec(f"individual = {strategy}({individual}, {scaling})")
            refined_population.append(individual)

        # Evaluate the fitness of each individual in the refined population
        for individual in refined_population:
            exec(f"individual = {func}({individual})")

        # Return the individual with the best fitness
        best_individual = min(refined_population, key=lambda x: exec(f"{func}({x})"))
        return best_individual

# Test the algorithm
def f1(x):
    return np.sum(x**2)

def f2(x):
    return np.sum(x**3)

def f3(x):
    return np.sum(x**4)

def f4(x):
    return np.sum(x**5)

def f5(x):
    return np.sum(x**6)

def f6(x):
    return np.sum(x**7)

def f7(x):
    return np.sum(x**8)

def f8(x):
    return np.sum(x**9)

def f9(x):
    return np.sum(x**10)

def f10(x):
    return np.sum(x**11)

def f11(x):
    return np.sum(x**12)

def f12(x):
    return np.sum(x**13)

def f13(x):
    return np.sum(x**14)

def f14(x):
    return np.sum(x**15)

def f15(x):
    return np.sum(x**16)

def f16(x):
    return np.sum(x**17)

def f17(x):
    return np.sum(x**18)

def f18(x):
    return np.sum(x**19)

def f19(x):
    return np.sum(x**20)

def f20(x):
    return np.sum(x**21)

def f21(x):
    return np.sum(x**22)

def f22(x):
    return np.sum(x**23)

def f23(x):
    return np.sum(x**24)

def f24(x):
    return np.sum(x**25)

algorithm = HybridEvolutionaryAlgorithm(10, 2)
print(algorithm(f1))