import random
import numpy as np

class Metaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func):
        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(func(self.search_space)))
        func_values = [func(x) for x in random.sample(self.search_space, num_evals)]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func]

        return best_func

class MutationMetaheuristic(Metaheuristic):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def mutate(self, individual):
        # Refine the strategy by changing the individual's line of the selected solution
        if random.random() < 0.45:
            line = random.randint(1, self.dim)
            individual[line] = random.uniform(-5.0, 5.0)

        return individual

class SelectionMetaheuristic(Metaheuristic):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def select(self, individuals):
        # Select the best individual based on the probability 0.45
        return [individual for individual in individuals if random.random() < 0.45]

class OptimizationAlgorithm:
    def __init__(self, budget, dim):
        self.mutation_metaheuristic = MutationMetaheuristic(budget, dim)
        self.selection_metaheuristic = SelectionMetaheuristic(budget, dim)

    def __call__(self, func):
        # Optimize the function using the metaheuristic algorithm
        individuals = [self.mutation_metaheuristic() for _ in range(100)]  # Run 100 iterations
        selected_individuals = self.selection_metaheuristic([individuals])

        # Evaluate the best individual
        best_func = max(set(func(selected_individuals), key=func(selected_individuals).count))
        best_func_values = [func(x) for x in selected_individuals]

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func_values]

        return best_func

# Example usage
budget = 1000
dim = 5
algorithm = OptimizationAlgorithm(budget, dim)
best_func = algorithm(__call__(lambda x: x**2))
print(f"Best function: {best_func}")