import random
import numpy as np

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func, budget=100, steps=1000):
        """
        Novel Metaheuristic Algorithm for Black Box Optimization (BMBO)

        Parameters:
        func (function): Black box function to optimize
        budget (int): Number of function evaluations
        steps (int): Number of iterations in the metaheuristic algorithm

        Returns:
        individual (array): Optimized individual
        """
        # Initialize population with random individuals
        population = [self.generate_individual() for _ in range(1000)]

        # Run the metaheuristic algorithm
        for _ in range(steps):
            # Evaluate fitness of each individual
            fitnesses = [self.evaluate_fitness(individual, func, self.budget) for individual in population]

            # Select parents using tournament selection
            parents = self.select_parents(population, fitnesses, budget)

            # Crossover (reproduce) parents to create offspring
            offspring = self.crossover(parents)

            # Mutate offspring to introduce genetic variation
            offspring = self.mutate(offspring)

            # Replace least fit individuals with new offspring
            population = self.replace_least_fit(population, offspring, fitnesses)

        # Return the fittest individual
        return population[0]

    def generate_individual(self):
        """
        Generate a random individual within the search space

        Returns:
        array: Optimized individual
        """
        return np.random.uniform(self.search_space[:, 0], self.search_space[:, 1], size=(1, 2))

    def select_parents(self, population, fitnesses, budget):
        """
        Select parents using tournament selection

        Parameters:
        population (array): List of individuals
        fitnesses (array): Fitness values of each individual
        budget (int): Number of function evaluations

        Returns:
        array: Parents
        """
        parents = []
        for _ in range(budget):
            parent = random.choice(population)
            fitness = fitnesses[np.random.randint(0, len(fitnesses))]
            if np.linalg.norm(parent - self.search_space[np.random.randint(0, self.search_space.shape[0])]) < fitness:
                parents.append(parent)
        return parents

    def crossover(self, parents):
        """
        Perform crossover (reproduce) on parents to create offspring

        Parameters:
        parents (array): List of parents

        Returns:
        array: Offspring
        """
        offspring = []
        for _ in range(len(parents)):
            parent1, parent2 = random.sample(parents, 2)
            offspring.append(np.vstack((parent1, parent2)))
        return offspring

    def mutate(self, offspring):
        """
        Mutate offspring to introduce genetic variation

        Parameters:
        offspring (array): Offspring

        Returns:
        array: Mutated offspring
        """
        mutated_offspring = []
        for individual in offspring:
            mutated_individual = individual.copy()
            np.random.shuffle(mutated_individual)
            mutated_offspring.append(mutated_individual)
        return mutated_offspring

    def replace_least_fit(self, population, offspring, fitnesses):
        """
        Replace least fit individuals with new offspring

        Parameters:
        population (array): List of individuals
        offspring (array): Offspring
        fitnesses (array): Fitness values of each individual

        Returns:
        array: Population with least fit individuals replaced
        """
        population = np.delete(population, np.argmin(fitnesses), axis=0)
        return np.vstack((population, offspring))