import numpy as np

class DifferentialEvolutionWithAdaptiveCrossover:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 100
        self.num_particles = self.population_size
        self.num_iterations = self.budget
        self.crossover_probability = 0.5
        self.adaptation_rate = 0.1
        self.f = None

    def __call__(self, func):
        self.f = func
        for _ in range(self.num_iterations):
            # Initialize population
            population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

            # Main loop
            for _ in range(self.num_iterations):
                # Evaluate population
                values = self.f(population)

                # Update population
                new_population = population.copy()
                for i in range(self.population_size):
                    j = np.random.randint(0, self.population_size - 1)
                    k = np.random.randint(0, self.population_size - 1)

                    # Differential evolution
                    differential = population[j, :] - population[k, :]
                    mutated = population[i, :] + differential * np.random.uniform(-1.0, 1.0, self.dim)

                    # Adaptive crossover
                    if random.random() < self.crossover_probability:
                        child = (mutated + population[i, :]) / 2
                    else:
                        child = mutated

                    # Replace individual
                    new_population[i, :] = child

                # Replace population
                population = new_population

                # Check for termination
                if np.all(values <= np.min(values)):
                    break

        # Return the best solution
        return np.min(values)

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

optimizer = DifferentialEvolutionWithAdaptiveCrossover(budget=100, dim=2)
result = optimizer(func)
print(result)