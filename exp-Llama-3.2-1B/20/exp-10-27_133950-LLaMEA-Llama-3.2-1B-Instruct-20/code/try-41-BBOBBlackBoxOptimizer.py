import numpy as np
from scipy.optimize import minimize, differential_evolution

class BBOBBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0

    def __call__(self, func):
        def wrapper(x):
            if self.func_evaluations >= self.budget:
                raise Exception("Optimization limit reached")
            return func(x)

        self.func_evaluations += 1
        try:
            result = minimize(wrapper, self.search_space[0], method="SLSQP", bounds=[(x, x) for x in self.search_space])
            return result.x
        except Exception as e:
            print(f"Error: {e}")
            return None

    def hybrid_search(self, population_size, mutation_rate, mutation_threshold, selection_rate, num_generations):
        # Initialize population with random individuals
        population = [np.random.uniform(self.search_space) for _ in range(population_size)]

        for generation in range(num_generations):
            # Select parents using roulette wheel selection
            parent_selection = np.random.choice(len(population), size=population_size, p=[1 - selection_rate, selection_rate])
            parent_indices = np.array(parent_selection)

            # Perform crossover and mutation on selected parents
            children = []
            for parent_index in parent_indices:
                parent = population[parent_index]
                child = parent.copy()
                if np.random.rand() < mutation_rate:
                    child[np.random.randint(0, self.dim)] += np.random.uniform(-mutation_threshold, mutation_threshold)
                children.append(child)

            # Evaluate fitness of each child
            child_evaluations = [self(func, child) for func, child in zip(self.func, children)]

            # Select parents with best fitness
            parent_indices = np.argsort(child_evaluations)[:int(population_size/2)]
            population = [population[parent_index] for parent_index in parent_indices]

        # Optimize the best individual
        best_individual = self(func, population[0])
        best_fitness = self.func(best_individual)

        return best_individual, best_fitness

# Example usage:
optimizer = BBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
best_individual, best_fitness = optimizer.hybrid_search(population_size=100, mutation_rate=0.1, mutation_threshold=0.5, selection_rate=0.5, num_generations=100)
print(f"Best individual: {best_individual}, Best fitness: {best_fitness}")