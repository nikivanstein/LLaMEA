import numpy as np

class PHEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = self.initialize_population()
        self.score = np.inf

    def initialize_population(self):
        population = []
        for _ in range(self.budget):
            # Randomly select a dimension
            dim = np.random.randint(0, self.dim)
            # Randomly generate a solution within the search space
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(solution)
        return population

    def evaluate(self, func):
        # Evaluate the fitness of each solution
        fitness = np.array([func(solution) for solution in self.population])
        # Update the population with the fittest solutions
        self.population = self.population[np.argsort(fitness)][-self.budget:]
        # Update the score
        self.score = np.min(fitness)

    def refine_solution(self, index):
        # Select a random solution to refine
        solution = self.population[np.random.randint(0, self.budget)]
        # Randomly select a dimension to change
        dim = np.random.randint(0, self.dim)
        # Randomly select a change type (within the bounds of the search space)
        change_type = np.random.choice(['+/-', '-/+', '*/+'])
        # Apply the change
        if change_type == '+/-':
            solution[dim] += np.random.uniform(-1.0, 1.0)
        elif change_type == '-/+':
            solution[dim] -= np.random.uniform(1.0, 5.0)
        elif change_type == '*/+':
            solution[dim] *= np.random.uniform(0.5, 2.0)
        # Return the refined solution
        return solution

    def __call__(self, func):
        # Evaluate the initial population
        self.evaluate(func)
        # Refine the solution using probability 0.3
        for _ in range(int(self.budget * 0.3)):
            index = np.random.randint(0, self.budget)
            self.population[index] = self.refine_solution(index)
        # Evaluate the refined population
        self.evaluate(func)