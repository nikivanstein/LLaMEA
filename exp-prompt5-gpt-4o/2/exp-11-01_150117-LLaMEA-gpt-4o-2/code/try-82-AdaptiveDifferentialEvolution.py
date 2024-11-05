import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # A common heuristic is to use 10 times the dimension
        self.population = np.random.uniform(-5, 5, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.F = 0.7  # Changed line: Increased mutation factor from 0.65 to 0.7
        self.CR = 0.9  # Initial crossover probability
        self.evaluations = 0

    def __call__(self, func):
        initial_budget = self.budget  # Store the initial budget for cooling schedule
        no_improvement_count = 0  # Added line: Count the number of iterations without improvement

        while self.evaluations < self.budget:
            # Evaluate initial population
            for i in range(self.population_size):
                if self.fitness[i] == np.inf:
                    self.fitness[i] = func(self.population[i])
                    self.evaluations += 1

            # Main loop
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                # Mutation: select three random individuals
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]

                # Differential Mutation
                mutant = np.clip(a + self.F * (b - c), -5, 5)

                # Crossover
                trial = np.copy(self.population[i])
                jrand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == jrand:
                        trial[j] = mutant[j]

                # Selection
                trial_fitness = func(trial)
                self.evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    no_improvement_count = 0  # Reset the counter on improvement
                    
                    # Adapt F and CR dynamically based on success
                    self.F = min(1.0, self.F + 0.1)
                    self.CR = min(1.0, self.CR + 0.35)  # Changed line: Increase CR increment from 0.25 to 0.35
                else:
                    # Slightly decrease F and CR if not successful
                    self.F = max(0.05, self.F - 0.03)  # Changed line: Adjust mutation factor decrement (from 0.02 to 0.03)
                    self.CR = max(0.1, self.CR - 0.01)
                    no_improvement_count += 1  # Increment the counter if no improvement

            # Dynamically adjust population size based on fitness variance
            if np.var(self.fitness) < 1e-4:
                self.population_size = max(4, int(self.population_size * 0.75))  # Changed line: More gradual reduction
                self.population = np.resize(self.population, (self.population_size, self.dim))
                self.fitness = np.resize(self.fitness, self.population_size)

            # Cooling schedule for mutation factor F
            self.F *= 0.99  # Changed line: Gradually reduce F over iterations

            # Random restart mechanism
            if no_improvement_count > 20:  # If no improvement for 20 iterations
                self.population = np.random.uniform(-5, 5, (self.population_size, self.dim))
                self.fitness.fill(np.inf)
                no_improvement_count = 0  # Reset the counter

        # Return the best found solution
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]