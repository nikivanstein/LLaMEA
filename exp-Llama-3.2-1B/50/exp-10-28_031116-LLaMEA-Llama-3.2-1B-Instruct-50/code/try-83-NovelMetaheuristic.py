import numpy as np

class NovelMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []

    def __call__(self, func):
        # Initialize population with random solutions
        for _ in range(100):
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            # Evaluate function at each solution
            func(solution)
            # Add solution to population if it hasn't exceeded budget
            if len(self.population) < self.budget:
                self.population.append(solution)
            else:
                # Refine strategy to increase chances of finding a better solution
                # based on the history of previous solutions
                if len(self.population) > 10:
                    solution, _ = self.population[np.random.choice(len(self.population), 1)]
                # Use probability 0.45 to change individual lines of the solution
                if np.random.rand() < 0.45:
                    solution = solution + np.random.uniform(-0.1, 0.1, self.dim)
                self.population[np.random.choice(len(self.population))] = solution
        # Evaluate final solution
        func(self.population[-1])

# BBOB test suite functions
def func1(x):
    return np.sum(x)

def func2(x):
    return np.prod(x)

def func3(x):
    return np.mean(x)

def func4(x):
    return np.max(x)

# Example usage
novel_metaheuristic = NovelMetaheuristic(100, 5)
novel_metaheuristic(func1)
novel_metaheuristic(func2)
novel_metaheuristic(func3)
novel_metaheuristic(func4)