import numpy as np

class Enhanced_PSO_SA_Optimizer:
    def __init__(self, budget, dim, swarm_size=30, max_iter=1000, c1=2.0, c2=2.0, initial_temp=10.0, cooling_rate=0.95):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.c1 = c1
        self.c2 = c2
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate

    def __call__(self, func):
        swarm_position = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
        swarm_velocity = np.zeros((self.swarm_size, self.dim))
        global_best_position = swarm_position[np.argmin(np.apply_along_axis(func, 1, swarm_position))]
        global_best_value = func(global_best_position)
        temperature = self.initial_temp

        for _ in range(self.max_iter):
            r1, r2 = np.random.rand(self.swarm_size, 1), np.random.rand(self.swarm_size, 1)
            swarm_velocity = 0.3 * swarm_velocity + self.c1 * r1 * (global_best_position - swarm_position) + self.c2 * r2 * (global_best_position - swarm_position)
            swarm_position = np.clip(swarm_position + swarm_velocity, -5.0, 5.0)
            fitness_values = np.apply_along_axis(func, 1, swarm_position)
            best_idx = np.argmin(fitness_values)
            if fitness_values[best_idx] < global_best_value:
                global_best_value = fitness_values[best_idx]
                global_best_position = np.copy(swarm_position[best_idx])
            acceptance_prob = np.exp((global_best_value - fitness_values) / temperature)
            accept_mask = np.random.rand(self.swarm_size) < acceptance_prob.flatten()
            swarm_position[accept_mask] = np.clip(swarm_position[accept_mask], -5.0, 5.0)
            temperature *= self.cooling_rate

        return global_best_position