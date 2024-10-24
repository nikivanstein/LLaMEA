import numpy as np

class MetaHeuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, dim))

    def __call__(self, func):
        def eval_func(x):
            return func(x)

        def evaluate_func(x):
            return np.min(np.abs(eval_func(x)))

        def fitness(x):
            return evaluate_func(x) / self.budget

        # Initialize the population with random solutions
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        # Run the genetic algorithm for the specified number of generations
        for _ in range(100):
            # Calculate the fitness of each individual
            fitnesses = fitness(self.population)

            # Select the fittest individuals using tournament selection
            tournament_size = self.population_size // 2
            tournament_selections = np.random.choice(fitnesses, size=(tournament_size, tournament_size), replace=False)
            tournament_fitnesses = np.min(np.abs(np.array(tournament_selections) / fitnesses))

            # Create a new population by combining the fittest individuals
            new_population = np.concatenate([self.population[np.argsort(tournament_fitnesses)], self.population[np.argsort(tournament_fitnesses)[~np.argsort(tournament_fitnesses)]]], axis=0)

            # Update the population with the new generation
            self.population = new_population

            # Prune the population to maintain the specified population size
            self.population = self.population[:self.population_size]

# Example usage:
# Create a new MetaHeuristic instance with a budget of 100 evaluations and a dimension of 10
metaheuristic = MetaHeuristic(100, 10)

# Optimize the black box function f(x) = x^2 using the MetaHeuristic algorithm
def f(x):
    return x**2

# Evaluate the function 100 times to get a score
metaheuristic(func=f)

# Print the current population and score
print("Current population:", metaheuristic.population)
print("Score:", metaheuristic.score)