import numpy as np
from scipy.optimize import differential_evolution
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

class MLEAdH:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.elitism_ratio = 0.2
        self.num_iterations = 10
        self.num_hidden_layers = 2
        self.num_hidden_units = 10

    def __call__(self, func):
        # Initialize population with random points
        population = np.random.uniform(self.search_space[0], self.search_space[1], size=(self.budget, self.dim))

        # Initialize elite set
        elite_set = population[:int(self.budget * self.elitism_ratio)]

        # Perform meta-learning
        for _ in range(self.num_iterations):
            # Split population into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(population, np.array([func(x) for x in population]), test_size=0.2, random_state=42)

            # Train neural network
            model = MLPRegressor(hidden_layer_sizes=(self.num_hidden_units,) * self.num_hidden_layers, activation='relu', solver='adam', max_iter=self.num_iterations)
            model.fit(X_train, y_train)

            # Evaluate model
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            print(f"MSE after iteration {_+1}: {mse}")

        # Perform differential evolution
        for _ in range(self.budget - len(elite_set)):
            # Evaluate population
            fitness = np.array([func(x) for x in population])

            # Perform differential evolution
            new_population = differential_evolution(func, self.search_space, x0=elite_set, popsize=len(elite_set) + 1, maxiter=1)

            # Update population and elite set
            population = np.concatenate((elite_set, new_population[0:1]))
            elite_set = population[:int(self.budget * self.elitism_ratio)]

        # Return the best solution
        return np.min(func(population))

# Example usage:
def func(x):
    return np.sum(x**2)

ml_e_adh = MLEAdH(budget=100, dim=10)
best_solution = ml_e_adh(func)
print(f"Best solution: {best_solution}")