import numpy as np

class DEAED:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.f_best = None
        self.x_best = None
        self.f_best_history = []
        self.x_best_history = []
        self.refine_probability = 0.05

    def __call__(self, func):
        if self.budget <= 0:
            return self.f_best, self.x_best

        # Initialize population
        population = np.random.uniform(-5.0, 5.0, (self.dim, self.budget))
        population = population / np.max(np.abs(population))

        for _ in range(self.budget):
            # Evaluate population
            f_values = func(population)

            # Calculate fitness
            fitness = np.array(f_values) - np.min(f_values, axis=1, keepdims=True)
            fitness = 1 / (1 + fitness)

            # Update best solution
            if self.f_best is None or np.mean(fitness) > np.mean(self.f_best):
                self.f_best = fitness
                self.x_best = population[np.argmin(fitness, axis=1)]

            # Update history
            self.f_best_history.append(self.f_best)
            self.x_best_history.append(self.x_best)

            # Differential evolution
            for i in range(self.dim):
                for j in range(1, self.budget // self.dim):
                    # Calculate crossover point
                    k = np.random.randint(j * self.dim, (j + 1) * self.dim)

                    # Calculate differential evolution vector
                    diff = population[k, :] - population[i, :]
                    diff = diff / np.max(np.abs(diff))

                    # Calculate new solution
                    new_solution = population[i, :] + diff
                    new_solution = new_solution / np.max(np.abs(new_solution))

                    # Evaluate new solution
                    f_value = func(new_solution)

                    # Update best solution
                    if f_value < self.f_best[i]:
                        self.f_best[i] = f_value
                        self.x_best[i] = new_solution

                    # Refine individual lines with probability 0.05
                    if np.random.rand() < self.refine_probability:
                        # Randomly select two individuals
                        idx1 = np.random.randint(0, self.dim)
                        idx2 = np.random.randint(0, self.dim)

                        # Calculate crossover point
                        k = np.random.randint(idx1 + 1, idx2)

                        # Calculate new solution
                        new_solution = population[idx1, :] + (population[idx2, :] - population[idx1, :]) * np.random.uniform(0, 1)

                        # Evaluate new solution
                        f_value = func(new_solution)

                        # Update best solution
                        if f_value < self.f_best[idx1]:
                            self.f_best[idx1] = f_value
                            self.x_best[idx1] = new_solution

        return self.f_best, self.x_best

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
deaed = DEAED(budget, dim)
f_best, x_best = deaed(func)
print("Best fitness:", f_best)
print("Best solution:", x_best)