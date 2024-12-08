import numpy as np

class FlockSearch:
    def __init__(self, budget, dim, w=0.5, c1=1.5, c2=1.5):
        self.budget = budget
        self.dim = dim
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, (self.budget, self.dim))

        def update_position(position, velocity):
            new_position = position + velocity
            new_position = np.clip(new_position, -5.0, 5.0)
            return new_position

        def update_velocity(velocity, best_position, global_best_position):
            inertia = self.w * velocity
            cognitive = self.c1 * np.random.rand() * (best_position - velocity)
            social = self.c2 * np.random.rand() * (global_best_position - velocity)
            new_velocity = inertia + cognitive + social
            return new_velocity

        population = initialize_population()
        velocity = np.zeros((self.budget, self.dim))
        best_position = population[np.argmin(func(population))]
        global_best_position = np.copy(best_position)
        
        for _ in range(self.budget):
            for i in range(self.budget):
                velocity[i] = update_velocity(velocity[i], population[i], global_best_position)
                population[i] = update_position(population[i], velocity[i])
                if func(population[i]) < func(best_position):
                    best_position = population[i]
                if func(population[i]) < func(global_best_position):
                    global_best_position = population[i]

        return best_position