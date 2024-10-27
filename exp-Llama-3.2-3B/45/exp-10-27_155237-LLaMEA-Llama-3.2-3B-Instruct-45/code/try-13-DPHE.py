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

# Example usage:
if __name__ == "__main__":
    # Define a sample black box function
    def func(x):
        return np.sum(x**2)

    # Initialize the DPHE algorithm
    dphe = DPHE(budget=100, dim=10)

    # Optimize the function
    result = dphe(func)

    # Print the result
    if result is not None:
        print("Optimal solution:", result)
    else:
        print("Failed to converge")

# Novel "Differential Perturbation and Hybrid Evolution" (DPHE) algorithm with probability-based mutation strategy
# ```python
import numpy as np
from scipy.optimize import differential_evolution

class DPHE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.probability = 0.45

    def __call__(self, func):
        def neg_func(x):
            return -func(x)

        bounds = [(self.lower_bound, self.upper_bound) for _ in range(self.dim)]

        def mutate(x):
            mutated_x = x.copy()
            for i in range(self.dim):
                if np.random.rand() < self.probability:
                    mutated_x[i] += np.random.uniform(-1, 1)
            return mutated_x

        def evaluate_fitness(individual):
            return neg_func(individual)

        def evaluateBBOB(individual):
            # Evaluate the function using the DPHE algorithm
            x = individual
            f = func(x)
            aucs = np.array([f])
            np.save(f"currentexp/aucs-{self.__class__.__name__}-{i}.npy", aucs)
            return f

        def hybridEvolution(individual):
            # Hybrid evolution using differential evolution and mutation
            x = individual
            f = evaluate_fitness(x)
            mutated_x = mutate(x)
            f_mutated = evaluate_fitness(mutated_x)
            if f_mutated < f:
                return mutated_x
            else:
                return x

        # Initialize the population with random individuals
        population = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.budget, self.dim))

        # Evolve the population using hybrid evolution
        for i in range(self.budget):
            # Select the fittest individual
            fittest_individual = population[np.argmin([evaluate_fitness(individual) for individual in population])]
            # Mutate the fittest individual
            mutated_individual = hybridEvolution(fittest_individual)
            # Replace the fittest individual with the mutated individual
            population[i] = mutated_individual

        # Return the fittest individual
        return np.min(population, axis=0)

# Example usage:
if __name__ == "__main__":
    # Define a sample black box function
    def func(x):
        return np.sum(x**2)

    # Initialize the DPHE algorithm
    dphe = DPHE(budget=100, dim=10)

    # Optimize the function
    result = dphe(func)

    # Print the result
    if result is not None:
        print("Optimal solution:", result)
    else:
        print("Failed to converge")