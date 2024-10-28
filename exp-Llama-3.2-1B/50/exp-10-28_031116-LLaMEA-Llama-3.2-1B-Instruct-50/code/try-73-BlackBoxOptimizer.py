import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []

    def __call__(self, func):
        # Initialize the population with random solutions
        for _ in range(self.budget):
            self.population.append(self._generate_solution(func))

        # Evaluate the population and select the best individual
        best_individual = self._select_best_individual()

        # Optimize the best individual using the selected strategy
        return self._optimize_best_individual(best_individual)

    def _generate_solution(self, func):
        # Generate a random solution within the search space
        return np.random.uniform(-5.0, 5.0, self.dim)

    def _select_best_individual(self):
        # Select the best individual based on the probability distribution
        probabilities = np.zeros(self.budget)
        for i, individual in enumerate(self.population):
            probabilities[i] = np.random.rand()  # Randomly select a probability
        return self.population[np.argmax(probabilities)]

    def _optimize_best_individual(self, individual):
        # Optimize the individual using the selected strategy
        # For this example, we'll use the 0.45 probability to refine the strategy
        refined_individual = individual
        for _ in range(10):  # Refine the strategy 10 times
            # Use the 0.45 probability to change the individual
            if np.random.rand() < 0.45:
                refined_individual = self._refine_individual(refined_individual)
            # Use the 0.55 probability to change the individual
            else:
                refined_individual = self._refine_individual(refined_individual)
        return refined_individual