import numpy as np

class NovelMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func):
        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(func(self.search_space)))
        func_values = [func(x) for x in np.random.choice(self.search_space, num_evals, replace=False)]

        # Select the best function value
        best_func = np.argmax(set(func_values))

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func]

        return best_func

    def mutate(self, individual):
        # Refine the strategy by changing the individual's lines
        num_mutations = min(self.budget, len(individual) // 4)
        mutated_individuals = []
        for _ in range(num_mutations):
            mutated_individual = individual.copy()
            mutation_index = np.random.randint(0, len(individual))
            mutated_individual[mutation_index] = random.uniform(-5.0, 5.0)
            mutated_individuals.append(mutated_individual)

        return mutated_individuals

# Initialize the algorithm
algorithm = NovelMetaheuristic(budget=100, dim=10)

# Run the optimization
result = algorithm(__call__, func)

# Print the result
print("Optimal function:", result)
print("Optimal function value:", np.max(set(func_values)))
print("Best individual:", algorithm.__call__(func))
print("Best function value:", np.max(set(func_values)))