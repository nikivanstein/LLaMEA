import numpy as np
import random

class DifferentialEvolutionWithAdaptiveLearningRate:
    def __init__(self, budget, dim, alpha=0.3, beta=0.8, gamma=0.7):
        """
        Initialize the Differential Evolution with Adaptive Learning Rate algorithm.

        Parameters:
        budget (int): The maximum number of function evaluations.
        dim (int): The dimensionality of the search space.
        alpha (float, optional): The learning rate for the differential evolution. Defaults to 0.3.
        beta (float, optional): The mutation probability. Defaults to 0.8.
        gamma (float, optional): The adaptive learning rate. Defaults to 0.7.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.learning_rate = alpha

    def __call__(self, func):
        """
        Optimize the black box function using the Differential Evolution with Adaptive Learning Rate algorithm.

        Parameters:
        func (function): The black box function to optimize.

        Returns:
        float: The optimized function value.
        """
        while self.func_evaluations < self.budget:
            # Initialize the new individual
            new_individual = None
            # Randomly select the initial population
            population = [np.random.uniform(self.search_space) for _ in range(100)]
            # Evaluate the fitness of the initial population
            fitness_values = [self.evaluate_fitness(population[i], func) for i in range(100)]
            # Select the fittest individuals
            selected_indices = np.argsort(fitness_values)[-10:]
            # Select the new population
            new_population = population[selected_indices]
            # Update the new individual
            new_individual = self.update_individual(new_population, fitness_values, func)
            # Evaluate the fitness of the new population
            fitness_values = [self.evaluate_fitness(new_individual, func) for new_individual in new_population]
            # Select the fittest individuals
            selected_indices = np.argsort(fitness_values)[-10:]
            # Select the new population
            new_population = new_population[selected_indices]
            # Update the new individual
            new_individual = self.update_individual(new_population, fitness_values, func)
            # Update the search space
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
            # Update the learning rate
            self.learning_rate = self.alpha * self.learning_rate
            # Update the number of function evaluations
            self.func_evaluations += 1
        return self.evaluate_fitness(new_individual, func)

    def update_individual(self, population, fitness_values, func):
        """
        Update the new individual using the Differential Evolution with Adaptive Learning Rate algorithm.

        Parameters:
        population (list): The current population.
        fitness_values (list): The fitness values of the population.
        func (function): The black box function to optimize.

        Returns:
        float: The updated individual.
        """
        # Initialize the new individual
        new_individual = None
        # Randomly select the mutation probability
        mutation_probability = self.beta
        # Randomly select the number of mutations
        num_mutations = int(self.gamma * fitness_values[-1])
        # Randomly select the mutated individuals
        mutated_individuals = random.sample(population, num_mutations)
        # Update the new individual
        for mutated_individual in mutated_individuals:
            # Generate the new individual
            new_individual = func(mutated_individual)
            # Check if the new individual is valid
            if np.isnan(new_individual) or np.isinf(new_individual):
                continue
            # Check if the new individual is within the search space
            if not np.isin(new_individual, self.search_space):
                continue
            # Update the new individual
            new_individual = mutated_individual
        # Return the updated individual
        return new_individual

    def evaluate_fitness(self, individual, func):
        """
        Evaluate the fitness of the individual using the black box function.

        Parameters:
        individual (float): The individual to evaluate.
        func (function): The black box function to optimize.

        Returns:
        float: The fitness value of the individual.
        """
        return func(individual)

# Example usage
if __name__ == "__main__":
    # Create an instance of the Differential Evolution with Adaptive Learning Rate algorithm
    algorithm = DifferentialEvolutionWithAdaptiveLearningRate(budget=100, dim=10)
    # Optimize the black box function
    func = lambda x: x**2
    optimized_function = algorithm(func)
    print(f"Optimized function: {optimized_function}")