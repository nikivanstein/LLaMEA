import random
import math

class DynamicAdaptiveGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = [random.uniform(-5.0, 5.0) for _ in range(self.population_size)]
        self.fitnesses = [0] * self.population_size

    def __call__(self, func):
        # Initialize the population
        for _ in range(self.budget):
            # Select the next individual based on the fitness and the dimension
            # Use a simple strategy: select the individual with the highest fitness
            next_individual = max(self.population, key=lambda x: self.fitnesses[x])

            # Evaluate the function at the next individual
            fitness = func(next_individual)

            # Update the fitness and the population
            self.fitnesses[self.population_size - 1] += fitness
            self.population[self.population_size - 1] = next_individual

            # Ensure the fitness stays within the bounds
            self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

        # Return the best individual
        return self.population[0]

    def select_next_individual(self):
        # Select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        # Refine the strategy by changing the individual lines of the selected solution
        # to refine its strategy
        # For example, if the individual is a line, try to move it towards the center of the line
        # to improve its fitness
        individual = self.population[0]
        if individual in [x for x in self.population if x!= individual]:
            # Calculate the vector from the center of the line to the current individual
            center_vector = [0, 0]
            for x in self.population:
                if x!= individual:
                    center_vector[0] += x[0] - individual[0]
                    center_vector[1] += x[1] - individual[1]

            # Move the individual towards the center of the line
            # Use a simple strategy: move the individual by a small amount in the direction of the center vector
            direction_vector = [center_vector[0] / math.sqrt(center_vector[0]**2 + center_vector[1]**2), center_vector[1] / math.sqrt(center_vector[0]**2 + center_vector[1]**2]]
            move_amount = 0.1
            new_individual = [individual[0] + move_amount * direction_vector[0], individual[1] + move_amount * direction_vector[1]]

            # Ensure the new individual stays within the bounds
            new_individual = [min(max(new_individual[0], -5.0), 5.0), min(max(new_individual[1], -5.0), 5.0)]

            # Update the population
            self.population[self.population_size - 1] = new_individual

        # Return the updated individual
        return new_individual

# One-line description: "Dynamic Adaptive Genetic Algorithm with Adaptive Sampling and Refining Strategy"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting, and refines its strategy by changing the individual lines of the selected solution
# to improve its fitness.