import numpy as np
import random

class EvolutionaryOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []
        self.fitnesses = []
        self.boundaries = np.linspace(-5.0, 5.0, 100)

    def __call__(self, func):
        for _ in range(self.budget):
            # Select parents using tournament selection
            parents = self.select_parents(func, self.boundaries)

            # Crossover (reproduce) offspring
            offspring = self.crossover(parents)

            # Mutate offspring
            offspring = self.mutate(offspring)

            # Evaluate offspring using the function
            fitness = self.evaluate_func(offspring, func)

            # Store fitness and offspring
            self.population.append((fitness, offspring))
            self.fitnesses.append(fitness)

    def select_parents(self, func, boundaries):
        # Select parents using tournament selection
        parent1 = random.choice(boundaries)
        parent2 = random.choice(boundaries)
        while parent1 == parent2:
            parent2 = random.choice(boundaries)
        return [parent1, parent2]

    def crossover(self, parents):
        # Crossover (reproduce) offspring
        offspring = []
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i+1]
            # Select crossover point
            crossover_point = random.randint(1, len(parent1) - 2)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            offspring.append(child1)
            offspring.append(child2)
        return offspring

    def mutate(self, offspring):
        # Mutate offspring
        mutated_offspring = []
        for i in range(0, len(offspring), 2):
            parent1, parent2 = offspring[i], offspring[i+1]
            # Select mutation point
            mutation_point = random.randint(0, len(parent1) - 2)
            mutated_offspring.append(parent1[:mutation_point] + np.random.uniform(-1, 1, len(parent1)) + parent2[mutation_point:])
        return mutated_offspring

    def evaluate_func(self, offspring, func):
        # Evaluate offspring using the function
        fitness = 0
        for child in offspring:
            fitness += func(child)
        return fitness

    def print_population(self):
        # Print population
        print("Population:")
        for fitness, offspring in zip(self.fitnesses, self.population):
            print(f"Fitness: {fitness}, Offspring: {offspring}")

# Example usage
if __name__ == "__main__":
    budget = 100
    dim = 10
    optimization = EvolutionaryOptimization(budget, dim)
    func = lambda x: np.sin(x)
    optimization(func)
    optimization.print_population()