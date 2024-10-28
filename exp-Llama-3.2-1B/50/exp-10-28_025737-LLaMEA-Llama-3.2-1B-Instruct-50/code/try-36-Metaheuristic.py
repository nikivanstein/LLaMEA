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

        # Initialize the new population
        self.population = [self.evaluate_fitness(x) for x in self.search_space]

        # Apply genetic operations
        for _ in range(self.budget // 10):
            # Select parents using tournament selection
            parents = [self.evaluate_fitness(x) for x in random.sample(self.search_space, 10)]

            # Apply crossover and mutation
            offspring = []
            for _ in range(self.dim):
                parent1, parent2 = random.sample(parents, 2)
                child = (0.5 * (parent1 + parent2)) % 1
                if random.random() < 0.5:
                    child += random.uniform(-1, 1)
                offspring.append(child)

            # Update the population
            self.population = [self.evaluate_fitness(x) for x in offspring]

        # Apply simulated annealing
        temperature = 1000
        for _ in range(self.budget // 10):
            # Calculate the probability of accepting the current solution
            prob_accept = np.exp((self.population - self.evaluate_fitness(self.population[-1])) / temperature)

            # Accept the new solution with probability 0.45
            if random.random() < prob_accept:
                self.population.append(self.evaluate_fitness(self.population[-1]))
            else:
                self.population.append(self.evaluate_fitness(np.random.uniform(self.search_space)))

        # Return the best solution found
        return self.population[-1]

    def evaluate_fitness(self, individual):
        # Evaluate the function at the given individual
        return individual

# Example usage
problem = Metaheuristic(100, 10)
best_func = problem(problem(problem(problem(problem(problem(problem(problem(problem(problem(problem(np.array([1, 2, 3, 4, 5]))))), problem(problem(np.array([1, 2, 3, 4, 5])))))))))
print("Best function:", best_func)