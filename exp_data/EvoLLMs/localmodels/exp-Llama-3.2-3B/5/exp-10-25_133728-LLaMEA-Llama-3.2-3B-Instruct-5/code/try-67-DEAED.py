import numpy as np

class DEAED:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.f_best = None
        self.x_best = None
        self.f_best_history = []
        self.x_best_history = []

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

            # Adaptive mutation probability
            mutation_prob = np.random.uniform(0.05, 0.95)
            if np.random.rand() < mutation_prob:
                # Select a random individual
                i = np.random.randint(0, self.dim)
                # Select a random point in the population
                j = np.random.randint(0, self.budget)
                # Calculate the mutation vector
                mutation_vector = population[j, :] - population[i, :]
                mutation_vector = mutation_vector / np.max(np.abs(mutation_vector))
                # Apply the mutation
                new_solution = population[i, :] + mutation_vector
                new_solution = new_solution / np.max(np.abs(new_solution))
                # Evaluate the new solution
                f_value = func(new_solution)
                # Update the best solution
                if f_value < self.f_best[i]:
                    self.f_best[i] = f_value
                    self.x_best[i] = new_solution

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