import numpy as np

class DifferentialEvolutionWithAdaptiveLearningRate:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 100
        self.num_particles = self.population_size
        self.num_iterations = self.budget
        self.crossover_probability = 0.8
        self.adaptation_rate = 0.1
        self.difficulty = 0.5
        self.fitness_history = []
        self.pbest = np.zeros((self.population_size, self.dim))
        self.gbest = np.zeros(self.dim)
        self.x0 = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def __call__(self, func):
        for _ in range(self.num_iterations):
            # Initialize particles
            self.x0 = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
            self.fitness_history = []
            self.pbest = np.copy(self.x0)
            self.gbest = np.copy(self.x0[0])

            # Main loop
            for _ in range(self.num_iterations):
                # Evaluate particles
                values = func(self.x0)

                # Update pbest and gbest
                for i in range(self.population_size):
                    if values[i] < self.pbest[i, 0]:
                        self.pbest[i, :] = self.x0[i, :]
                    if values[i] < self.gbest[0]:
                        self.gbest[:] = self.x0[i, :]

                # Differential evolution
                for i in range(self.population_size):
                    j1, j2 = np.random.choice(self.population_size, 2, replace=False)
                    if random.random() < self.crossover_probability:
                        # Select two particles
                        x1, x2 = self.x0[j1, :], self.x0[j2, :]
                        child = (x1 + x2) / 2 + np.random.uniform(-self.difficulty, self.difficulty, self.dim)
                        child = child.clip(self.lower_bound, self.upper_bound)
                        self.x0[i, :] = child

                # Adaptive learning rate
                if len(self.fitness_history) > 1:
                    self.difficulty = self.adaptation_rate * (1 - len(self.fitness_history) / (self.num_iterations + 1))
                self.fitness_history.append(values)

            # Return the best solution
            return self.gbest[0]

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

optimizer = DifferentialEvolutionWithAdaptiveLearningRate(budget=100, dim=2)
result = optimizer(func)
print(result)