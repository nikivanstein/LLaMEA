import numpy as np

class AdaptiveDE:
    def __init__(self, budget, dim, mutation_rate, mutation_threshold):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.mutation_rate = mutation_rate
        self.mutation_threshold = mutation_threshold
        self.population_size = 100
        self.population = np.random.uniform(self.search_space, size=(self.population_size, self.dim))

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Select parents using tournament selection
            parents = np.array([self.select_parents(func, self.population, self.dim) for _ in range(self.population_size // 2)])

            # Create offspring using crossover
            offspring = np.array([self.crossover(parents[0], parents[1]) for _ in range(self.population_size // 2)])

            # Apply mutation
            self.population = np.array([self.applyMutation(offspring, self.population, self.mutation_rate, self.mutation_threshold) for _ in range(self.population_size)])

            # Evaluate the new population
            self.func_evaluations += 1
            func_value = func(self.population)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break

        return func_value

    def select_parents(self, func, population, dim):
        # Select parents using tournament selection
        tournament_size = 3
        tournament_indices = np.random.choice(population.shape[0], tournament_size, replace=False)
        tournament_values = np.array([func(population[i]) for i in tournament_indices])
        selected_indices = np.argsort(tournament_values)[:tournament_size]
        selected_parents = population[selected_indices]
        return selected_parents

    def crossover(self, parent1, parent2):
        # Create offspring using crossover
        crossover_point = np.random.randint(1, self.dim)
        offspring = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        return offspring

    def applyMutation(self, offspring, population, mutation_rate, mutation_threshold):
        # Apply mutation
        mutated_offspring = np.copy(offspring)
        for i in range(self.population_size):
            if np.random.rand() < mutation_rate:
                mutated_offspring[i] += np.random.uniform(-mutation_threshold, mutation_threshold)
        return mutated_offspring

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

adaptive_de = AdaptiveDE(1000, 2, 0.1, 0.5)
print(adaptive_de(test_function))  # prints a random value between -10 and 10