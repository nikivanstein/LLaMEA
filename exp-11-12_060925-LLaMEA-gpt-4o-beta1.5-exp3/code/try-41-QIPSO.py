import numpy as np

class QIPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.population_size = 10 * dim
        self.alpha = 0.5  # inertia weight
        self.beta = 0.9   # cognitive coefficient
        self.gamma = 0.9  # social coefficient

    def __call__(self, func):
        # Initialize population
        position = self.lower_bound + np.random.rand(self.population_size, self.dim) * (self.upper_bound - self.lower_bound)
        velocity = np.zeros((self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, position)
        self.evaluations = self.population_size
        
        # Initialize personal and global bests
        pbest_position = np.copy(position)
        pbest_fitness = np.copy(fitness)
        gbest_idx = np.argmin(fitness)
        gbest_position = position[gbest_idx].copy()
        gbest_fitness = fitness[gbest_idx]

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                
                # Update velocity and position
                cognitive_component = self.beta * np.random.rand(self.dim) * (pbest_position[i] - position[i])
                social_component = self.gamma * np.random.rand(self.dim) * (gbest_position - position[i])
                velocity[i] = self.alpha * velocity[i] + cognitive_component + social_component
                
                # Quantum-inspired position update with superposition principle
                distance_to_best = np.linalg.norm(gbest_position - position[i])
                superposition_prob = np.exp(-distance_to_best)  # probability of 'quantum jump'
                if np.random.rand() < superposition_prob:
                    new_position = gbest_position + np.random.uniform(-1, 1, self.dim) * (position[i] - gbest_position)
                else:
                    new_position = position[i] + velocity[i]

                # Boundary check
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)

                # Evaluate new position
                new_fitness = func(new_position)
                self.evaluations += 1

                # Update personal best
                if new_fitness < pbest_fitness[i]:
                    pbest_position[i] = new_position
                    pbest_fitness[i] = new_fitness

                    # Update global best
                    if new_fitness < gbest_fitness:
                        gbest_position = new_position
                        gbest_fitness = new_fitness

            # Dynamic adaptation of parameters
            self.alpha = max(0.4, self.alpha * 0.99)
            self.beta = min(1.2, self.beta * 1.01)
            self.gamma = min(1.2, self.gamma * 1.01)

        return gbest_position, gbest_fitness