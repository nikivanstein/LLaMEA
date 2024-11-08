import numpy as np

class Improved_PSO_SA_Optimizer:
    def __init__(self, budget, dim, swarm_size=30, max_iter=1000, c1=2.0, c2=2.0, initial_temp=10.0, cooling_rate=0.95):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.c1 = c1
        self.c2 = c2
        self.alpha = 0.3
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate

    def __call__(self, func):
        def objective_function(x):
            return func(x)

        def within_bounds(x):
            return np.clip(x, -5.0, 5.0)

        # Initialize particle positions and velocities
        swarm_position = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
        swarm_velocity = np.zeros((self.swarm_size, self.dim))

        # Initialize global best position and value
        global_best_position = np.random.uniform(-5.0, 5.0, self.dim)
        global_best_value = np.inf

        # Initialize SA temperature
        temperature = self.initial_temp

        for _ in range(self.max_iter):
            for i in range(self.swarm_size):
                # Update particle velocity
                r1, r2 = np.random.rand(), np.random.rand()
                swarm_velocity[i] = self.alpha * swarm_velocity[i] + self.c1 * r1 * (global_best_position - swarm_position[i]) + self.c2 * r2 * (global_best_position - swarm_position[i])

                # Update particle position
                swarm_position[i] = within_bounds(swarm_position[i] + swarm_velocity[i])

                # Calculate fitness value
                fitness_value = objective_function(swarm_position[i])

                # Update global best
                if fitness_value < global_best_value:
                    global_best_value = fitness_value
                    global_best_position = np.copy(swarm_position[i])

                # Simulated Annealing
                if np.random.rand() < np.exp((global_best_value - fitness_value) / temperature):
                    swarm_position[i] = within_bounds(swarm_position[i])

            # Cool down temperature
            temperature *= self.cooling_rate

        return global_best_position