import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Reshape

class MetaLEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.candidates = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_candidate = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_fitness = np.inf
        self.meta_model = self.create_meta_model()

    def create_meta_model(self):
        input_layer = Input(shape=(self.dim,))
        hidden_layer = Dense(64, activation='relu')(input_layer)
        output_layer = Dense(1)(hidden_layer)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = func(self.candidates[:, 0])
            self.best_candidate = self.candidates[np.argmin(self.candidates[:, 0]), :]
            self.best_fitness = fitness

            # Meta-learning
            inputs = self.candidates[:, 0].reshape((-1, self.dim))
            outputs = np.array([func(x) for x in inputs])
            self.meta_model.fit(inputs, outputs, epochs=10, verbose=0)
            meta_weights = self.meta_model.get_weights()

            # Refine evolutionary strategy
            new_candidates = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
            for i in range(self.population_size):
                new_candidate = new_candidates[i, :]
                new_candidate = new_candidate + self.meta_model.predict(new_candidate.reshape(1, self.dim))[0]
                new_candidate = new_candidate + np.random.uniform(-0.1, 0.1, size=(1, self.dim))[0]
                new_candidate = new_candidate.flatten()
                new_candidates[i, :] = new_candidate

            # Selection
            self.candidates = new_candidates
            self.candidates = self.candidates[np.argsort(self.candidates[:, 0])]
            self.population_size = self.population_size // 2

            # Mutation
            self.candidates[np.random.choice(self.population_size, size=self.population_size, replace=False), :] += np.random.uniform(-0.1, 0.1, size=(self.population_size, self.dim))

            # Check if the best candidate is improved
            if self.best_fitness < func(self.best_candidate):
                self.candidates[np.argmin(self.candidates[:, 0]), :] = self.best_candidate

        return self.best_candidate, self.best_fitness

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

meta_LEA = MetaLEvolutionaryAlgorithm(budget=100, dim=2)
best_candidate, best_fitness = meta_LEA(func)
print(f"Best candidate: {best_candidate}, Best fitness: {best_fitness}")