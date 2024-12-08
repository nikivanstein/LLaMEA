import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

class HyperNNMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x_init = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.y_init = np.zeros(budget)
        self.scaler = StandardScaler()

    def __call__(self, func):
        for _ in range(self.budget):
            x = self.x_init[np.random.choice(self.x_init.shape[0])]
            y = func(x)
            self.y_init[_] = y
            self.x_init = np.concatenate((self.x_init[:_], [x], self.x_init[_:]))
        
        # Train the neural network
        self.scaler.fit(self.x_init)
        self.x_init = self.scaler.transform(self.x_init)
        self.mlp = MLPRegressor(max_iter=100, learning_rate_init=0.1, hidden_layer_sizes=(10,), random_state=42)
        self.mlp.fit(self.x_init, self.y_init)
        
        # Use the trained neural network to optimize the function
        x_opt = self.mlp.predict(np.linspace(-5.0, 5.0, 1000).reshape(-1, 1))
        return x_opt[np.argmin(np.abs(x_opt - func(x_opt)))]

# Initialize the metaheuristic with a budget of 100 and a dimension of 5
metaheuristic = HyperNNMetaheuristic(100, 5)

# Test the metaheuristic on the BBOB test suite
from blackbox_optimization import bbob_test_suite
bbob_test_suite(metaheuristic, 24)