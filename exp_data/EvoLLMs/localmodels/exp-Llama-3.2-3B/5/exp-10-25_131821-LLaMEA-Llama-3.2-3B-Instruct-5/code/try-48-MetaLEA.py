import numpy as np
from scipy.optimize import differential_evolution
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

class MetaLEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.elitism_ratio = 0.2
        self.max_epochs = 100
        self.batch_size = 32

    def __call__(self, func):
        # Initialize population with random points
        population = np.random.uniform(self.search_space[0], self.search_space[1], size=(self.budget, self.dim))

        # Initialize elite set
        elite_set = population[:int(self.budget * self.elitism_ratio)]

        # Define meta-learning model
        X = np.array([x for x in population]).reshape(-1, self.dim)
        y = np.array([func(x) for x in population])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = MLPRegressor(hidden_layer_sizes=(10,), max_iter=self.max_epochs, batch_size=self.batch_size, learning_rate_init=0.01)
        model.fit(X_train, y_train)

        # Perform meta-learning
        for _ in range(self.budget - len(elite_set)):
            # Evaluate population
            fitness = np.array([func(x) for x in population])

            # Perform meta-learning
            new_population = model.predict(X_test)

            # Update population and elite set
            new_population = np.array(new_population).reshape(-1, self.dim)
            population = np.concatenate((elite_set, new_population))

            # Evaluate new population
            fitness = np.array([func(x) for x in population])

            # Update elite set
            elite_set = population[:int(self.budget * self.elitism_ratio)]

        # Return the best solution
        return np.min(func(population))

# Example usage:
def func(x):
    return np.sum(x**2)

meta_lea = MetaLEA(budget=100, dim=10)
best_solution = meta_lea(func)
print(f"Best solution: {best_solution}")