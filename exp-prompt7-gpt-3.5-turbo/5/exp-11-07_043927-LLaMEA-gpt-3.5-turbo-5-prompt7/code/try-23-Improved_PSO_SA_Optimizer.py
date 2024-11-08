import numpy as np

class Improved_PSO_SA_Optimizer:
    def __init__(self, budget, dim, swarm_size=30, max_iter=1000, c1=2.0, c2=2.0, initial_temp=10.0, cooling_rate=0.95):
        self.budget, self.dim, self.swarm_size, self.max_iter, self.c1, self.c2, self.initial_temp, self.cooling_rate = budget, dim, swarm_size, max_iter, c1, c2, initial_temp, cooling_rate

    def __call__(self, func):
        def objective_function(x):
            return func(x)

        def within_bounds(x):
            return np.clip(x, -5.0, 5.0)

        swarm_position = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
        swarm_velocity = np.zeros((self.swarm_size, self.dim))
        global_best_position = np.random.uniform(-5.0, 5.0, self.dim)
        global_best_value = np.inf
        temperature = self.initial_temp

        for _ in range(self.max_iter):
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(2)
                swarm_velocity[i] = 0.3 * swarm_velocity[i] + self.c1 * r1 * (global_best_position - swarm_position[i]) + self.c2 * r2 * (global_best_position - swarm_position[i])
                swarm_position[i] = within_bounds(swarm_position[i] + swarm_velocity[i])
                fitness_value = objective_function(swarm_position[i])

                if fitness_value < global_best_value:
                    global_best_value, global_best_position = fitness_value, np.copy(swarm_position[i])

                if np.random.rand() < np.exp((global_best_value - fitness_value) / temperature):
                    swarm_position[i] = within_bounds(swarm_position[i])

            temperature *= self.cooling_rate

        return global_best_position