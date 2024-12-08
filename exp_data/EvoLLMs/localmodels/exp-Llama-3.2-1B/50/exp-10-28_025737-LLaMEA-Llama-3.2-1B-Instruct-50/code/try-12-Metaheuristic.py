# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
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
    def __call__(self, func, mutation_rate):
        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(func(self.search_space)))
        func_values = [func(x) for x in random.sample(self.search_space, num_evals)]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Update the search space
        new_individual = [x for x in self.search_space if x not in best_func]
        new_individual.extend([x + random.uniform(-5, 5) for x in best_func])
        new_individual = [x for x in new_individual if x not in new_individual[:self.dim]]
        new_individual = np.array(new_individual)

        # Apply mutation
        if random.random() < mutation_rate:
            new_individual[random.randint(0, self.dim - 1)] = random.uniform(-5, 5)

        # Evaluate the new individual
        new_func_values = [func(x) for x in new_individual]

        # Select the best new function value
        best_new_func = max(set(new_func_values), key=new_func_values.count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_new_func]

        return best_new_func

class SelectionMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(func(self.search_space)))
        func_values = [func(x) for x in random.sample(self.search_space, num_evals)]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func]

        return best_func

class GeneticAlgorithm(Metaheuristic):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.population_size = 100
        self.mutation_rate = 0.1
        self.selection_rate = 0.1
        self.crossover_rate = 0.5
        self.population = [self.__call__(func, self.budget) for _ in range(self.population_size)]

    def __call__(self, func):
        # Select the best individual
        self.population = [self.__call__(func, self.budget) for _ in range(self.population_size)]
        self.population = SelectionMetaheuristic(self.budget, self.dim).(__call__(self.population[0]))

        # Mutate the best individual
        self.population = [MutationMetaheuristic(self.budget, self.dim).(__call__(self.population[i], self.mutation_rate)) for i in range(self.population_size)]

        # Evaluate the fitness of the population
        fitness = [self.__call__(func, self.budget) for func in self.population]
        fitness.sort(reverse=True)

        # Select the fittest individuals
        self.population = [self.population[i] for i in range(self.population_size) if i < self.selection_rate * self.population_size]

        # Return the fittest individual
        return self.population[0]

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
# Novel Metaheuristic Algorithm for Black Box Optimization
# ```
# ```python
ga = GeneticAlgorithm(100, 5)
print(ga.__call__(np.sin(np.linspace(-5, 5, 100))))