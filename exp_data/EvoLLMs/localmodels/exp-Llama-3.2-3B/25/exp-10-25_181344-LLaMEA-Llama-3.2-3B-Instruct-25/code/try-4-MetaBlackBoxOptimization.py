import numpy as np
import random
import time

class MetaBlackBoxOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.candidates = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_candidate = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_fitness = np.inf
        self.meta_model = None

    def __call__(self, func):
        # Initialize the meta model
        if self.meta_model is None:
            self.meta_model = self._initialize_meta_model()

        for _ in range(self.budget):
            # Sample a subset of candidates
            subset_candidates = self.candidates[np.random.choice(self.population_size, size=10, replace=False)]

            # Evaluate the subset of candidates
            fitness_values = func(subset_candidates)

            # Update the meta model
            self.meta_model.update(fitness_values)

            # Sample a new candidate using the meta model
            new_candidate = self.meta_model.sample()

            # Evaluate the new candidate
            new_fitness = func(new_candidate)

            # Update the best candidate and fitness
            if new_fitness < self.best_fitness:
                self.best_candidate = new_candidate
                self.best_fitness = new_fitness

            # Apply a probability of 0.25 to change the individual lines of the selected solution
            if random.random() < 0.25:
                self.candidates[np.random.choice(self.population_size, size=1, replace=False), :] = new_candidate

        return self.best_candidate, self.best_fitness

    def _initialize_meta_model(self):
        # Initialize the meta model using a simple neural network
        import tensorflow as tf
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.dim, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

meta_MBO = MetaBlackBoxOptimization(budget=100, dim=2)
best_candidate, best_fitness = meta_MBO(func)
print(f"Best candidate: {best_candidate}, Best fitness: {best_fitness}")