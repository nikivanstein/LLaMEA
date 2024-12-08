import numpy as np
import random
import time

class MetaLearningBasedEvolutionarySwarmOptimization:
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

            # Swarm Intelligence
            for _ in range(10):
                new_candidate = np.random.uniform(-5.0, 5.0, self.dim)
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

            # Meta-Learning
            if self.meta_model is None:
                self.meta_model = self._train_meta_model(func, self.candidates)
            else:
                new_candidates = self._sample_candidates(func, self.candidates, self.meta_model)
                new_fitness = func(new_candidates[:, 0])
                if np.any(new_fitness < self.best_fitness):
                    self.best_candidate = new_candidates[np.argmin(new_fitness), :]
                    self.best_fitness = np.min(new_fitness)
                    self.candidates[np.argmin(self.candidates[:, 0]), :] = self.best_candidate

            # Check if the best candidate is improved
            if self.best_fitness < func(self.best_candidate):
                self.candidates[np.argmin(self.candidates[:, 0]), :] = self.best_candidate

        return self.best_candidate, self.best_fitness

    def _train_meta_model(self, func, candidates):
        # Train a meta-model to predict the fitness of new candidates
        X_train, y_train = self._prepare_data(candidates)
        model = self._train_model(X_train, y_train)
        return model

    def _sample_candidates(self, func, candidates, meta_model):
        # Sample new candidates using the meta-model
        X_new = self._prepare_data(candidates)
        y_new = meta_model.predict(X_new)
        new_candidates = self._generate_candidates(X_new, y_new)
        return new_candidates

    def _prepare_data(self, candidates):
        # Prepare the data for training the meta-model
        X = np.array([candidate for candidate in candidates])
        y = np.array([func(candidate) for candidate in candidates])
        return X, y

    def _train_model(self, X, y):
        # Train a simple neural network to predict the fitness
        from sklearn.neural_network import MLPRegressor
        model = MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000)
        model.fit(X, y)
        return model

    def _generate_candidates(self, X_new, y_new):
        # Generate new candidates using the meta-model
        new_candidates = np.array([X_new[i] + np.random.uniform(-0.1, 0.1, size=self.dim) for i in range(X_new.shape[0])])
        return new_candidates

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

meta_LESO = MetaLearningBasedEvolutionarySwarmOptimization(budget=100, dim=2)
best_candidate, best_fitness = meta_LESO(func)
print(f"Best candidate: {best_candidate}, Best fitness: {best_fitness}")