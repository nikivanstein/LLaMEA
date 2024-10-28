import numpy as np

class AdaptiveHybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim, population_size=100, mutation_rate=0.01, sampling_rate=0.5):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.sampling_rate = sampling_rate
        self.population = np.random.uniform(-5.0, 5.0, size=(population_size, dim))
        self.population_history = []

    def __call__(self, func):
        # Adaptive sampling strategy
        adaptive_func = self.adaptive_sampling(func)
        for _ in range(self.budget):
            # Evaluate the function with the current population
            best_func = np.argmax(self.population_history)
            # Update the population with the best function
            if np.any(best_func!= self.population_history[best_func]):
                self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            # Update the adaptive function with the current population
            if np.all(best_func == self.population_history[best_func]):
                break
        # Store the history of the function values
        self.population_history.append(func(self.population))
        return func(self.population)

    def adaptive_sampling(self, func):
        # Adaptive sampling strategy
        num_evaluations = 0
        while num_evaluations < self.budget:
            # Evaluate the function with the current population
            func_evals = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            best_func = func(func_evals)
            # Update the adaptive function with the current population
            if np.any(best_func!= func(func_evals)):
                func_evals = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            # Update the adaptive function with the best function
            if np.all(best_func == func(func_evals)):
                break
            num_evaluations += 1
        return func_evals

# One-line description with the main idea
# AdaptiveHybridEvolutionaryAlgorithm: A hybrid algorithm that combines adaptive sampling and evolutionary optimization to solve black box optimization problems.
# To refine the strategy, the individual lines of the selected solution can be modified to change the probability of selecting the best function.
# Refine the strategy to increase the Area over the convergence curve (AOCC)
def adaptiveHybridEvolutionaryAlgorithm_refined(self, budget, dim):
    # Adaptive sampling strategy
    adaptive_func = self.adaptive_sampling(self.adaptive_sampling)
    for _ in range(self.budget):
        # Evaluate the function with the current population
        best_func = np.argmax(self.population_history)
        # Update the population with the best function
        if np.any(best_func!= self.population_history[best_func]):
            self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        # Update the adaptive function with the current population
        if np.all(best_func == self.population_history[best_func]):
            break
    return self.population

# Code:
# ```python
# AdaptiveHybridEvolutionaryAlgorithm: A hybrid algorithm that combines adaptive sampling and evolutionary optimization to solve black box optimization problems.
# To refine the strategy, the individual lines of the selected solution can be modified to change the probability of selecting the best function.
# Refine the strategy to increase the Area over the convergence curve (AOCC)
adaptiveHybridEvolutionaryAlgorithm_refined = AdaptiveHybridEvolutionaryAlgorithm(
    budget=1000, dim=10, population_size=200, mutation_rate=0.01, sampling_rate=0.5
)

# Average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.06 with standard deviation 0.06.
# One-line description with the main idea
# AdaptiveHybridEvolutionaryAlgorithm: A hybrid algorithm that combines adaptive sampling and evolutionary optimization to solve black box optimization problems.
# To refine the strategy, the individual lines of the selected solution can be modified to change the probability of selecting the best function.
# Refine the strategy to increase the Area over the convergence curve (AOCC)
# Code: 
# ```python
# ```python
# ```python
adaptiveHybridEvolutionaryAlgorithm_refined(adaptiveHybridEvolutionaryAlgorithm_refined, 20, 20, 0.5)