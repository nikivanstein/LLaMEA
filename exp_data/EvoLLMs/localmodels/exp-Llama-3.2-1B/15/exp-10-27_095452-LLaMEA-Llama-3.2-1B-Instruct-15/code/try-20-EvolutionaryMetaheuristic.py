import numpy as np

class EvolutionaryMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

    def __call__(self, func, budget=100):
        # Initialize population with random solutions
        population = [func(np.random.uniform(-5.0, 5.0, self.dim)) for _ in range(100)]

        # Run evolutionary algorithm
        for _ in range(1000):
            # Select parents using tournament selection
            parents = np.array(population).tolist()
            for _ in range(self.budget):
                parent1, parent2 = np.random.choice(population, 2, replace=False)
                if np.max(parent1) > np.max(parent2):
                    parents[parents.index(parent1)] = parent2
            # Crossover and mutation
            for i in range(len(parents) // 2):
                x1, y1 = parents[i], parents[i + 1]
                x2, y2 = np.random.uniform(-5.0, 5.0, self.dim), np.random.uniform(-5.0, 5.0, self.dim)
                if np.max(y1) > np.max(y2):
                    parents[i], parents[i + 1] = parents[i + 1], parents[i]
                    parents[i], parents[i + 1] = x2, x1
                if np.random.rand() < 0.1:
                    parents[i], parents[i + 1] = parents[i + 1], parents[i]

        # Return best solution
        return max(population)

# One-line description:
# EvolutionaryMetaheuristic: A novel metaheuristic algorithm for solving black box optimization problems using evolutionary algorithms with adaptive search strategies.

class HyperCanny:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Grid search
            x_values = np.linspace(-5.0, 5.0, 100)
            y_values = func(x_values)
            grid = dict(zip(x_values, y_values))
            best_x, best_y = None, None
            for x, y in grid.items():
                if x < best_x or (x == best_x and y < best_y):
                    best_x, best_y = x, y
            # Random search
            random_x_values = np.random.uniform(-5.0, 5.0, self.dim)
            random_y_values = func(random_x_values)
            random_x_values = np.array([x for x, y in zip(random_x_values, random_y_values) if -5.0 <= x <= 5.0])
            random_y_values = np.array([y for x, y in zip(random_x_values, random_y_values) if -5.0 <= y <= 5.0])
            # Evolutionary algorithm
            self.func_evals += 1
            x_values = random_x_values
            y_values = random_y_values
            for _ in range(100):
                x_new = x_values + np.random.uniform(-0.1, 0.1, self.dim)
                y_new = y_values + np.random.uniform(-0.1, 0.1, self.dim)
                if -5.0 <= x_new <= 5.0 and -5.0 <= y_new <= 5.0:
                    x_values = x_new
                    y_values = y_new
                    break
            # Check if the new solution is better
            if np.max(y_values) > np.max(y_values + 0.1):
                best_x, best_y = x_values, y_values
        return best_x, best_y

# One-line description:
# HyperCanny: A novel metaheuristic algorithm for solving black box optimization problems using a combination of grid search, random search, and evolutionary algorithms with adaptive search strategies.