import numpy as np
import random

class MetaLearningGuidedEvolutionarySwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.candidates = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_candidate = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_fitness = np.inf
        self.meta_model = None

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = func(self.candidates[:, 0])
            self.best_candidate = self.candidates[np.argmin(self.candidates[:, 0]), :]
            self.best_fitness = fitness

            # Evolutionary Strategy
            self.candidates[np.random.choice(self.population_size, size=10, replace=False), :] = self.candidates[np.random.choice(self.population_size, size=10, replace=False), :] + \
                                                                                      self.candidates[np.random.choice(self.population_size, size=10, replace=False), :] * \
                                                                                      np.random.uniform(-0.1, 0.1, size=(10, self.dim))

            # Meta-Learning
            if self.meta_model is None:
                self.meta_model = self._meta_learning(func)
            new_candidate = self.meta_model.predict(self.candidates[:, 0])
            new_fitness = func(new_candidate)
            if new_fitness < self.best_fitness:
                self.best_candidate = new_candidate
                self.best_fitness = new_fitness
                self.candidates[np.argmin(self.candidates[:, 0]), :] = new_candidate

            # Selection
            self.candidates = self.candidates[np.argsort(self.candidates[:, 0])]
            self.population_size = self.population_size // 2

            # Mutation
            self.candidates[np.random.choice(self.population_size, size=self.population_size, replace=False), :] += np.random.uniform(-0.1, 0.1, size=(self.population_size, self.dim))

            # Check if the best candidate is improved
            if self.best_fitness < func(self.best_candidate):
                self.candidates[np.argmin(self.candidates[:, 0]), :] = self.best_candidate

        return self.best_candidate, self.best_fitness

    def _meta_learning(self, func):
        # Initialize meta-model using a simple neural network
        model = neural_network(input_dim=self.dim, output_dim=1)
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(self.candidates[:, 0], func(self.candidates[:, 0]), epochs=10, batch_size=10)
        return model

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

meta_ESO = MetaLearningGuidedEvolutionarySwarmOptimization(budget=100, dim=2)
best_candidate, best_fitness = meta_ESO(func)
print(f"Best candidate: {best_candidate}, Best fitness: {best_fitness}")