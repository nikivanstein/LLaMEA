import numpy as np

class AdaptiveStepSizeEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []
        self.fitness_scores = []

    def __call__(self, func):
        # Evaluate the function 10 times with adaptive step size
        for _ in range(10):
            for i in range(self.budget):
                # Generate a random step size
                step_size = np.random.uniform(-0.1, 0.1)
                # Evaluate the function with the current step size
                func_value = func(i / step_size, self.dim)
                # Store the fitness score
                self.fitness_scores.append(func_value)

        # Select the fittest individual
        self.population = sorted(self.fitness_scores, reverse=True)[:self.budget // 2]
        # Refine the step size based on the fitness scores
        self.step_size = np.mean(self.fitness_scores[:self.budget // 2])

    def mutate(self, individual):
        # Generate a new individual with a random step size
        new_individual = individual + np.random.uniform(-self.step_size, self.step_size)
        # Ensure the new individual stays within the search space
        new_individual = np.clip(new_individual, -5.0, 5.0)
        return new_individual

    def __str__(self):
        return f"Population: {self.population}, Fitness Scores: {self.fitness_scores}"

# Description: Evolutionary Algorithm with Adaptive Step Size and Mutation
# Code: 
# ```python
algorithm = AdaptiveStepSizeEvolutionaryAlgorithm(budget=100, dim=5)
func = lambda x: np.sin(x)
algorithm(__call__(func))
print(algorithm)