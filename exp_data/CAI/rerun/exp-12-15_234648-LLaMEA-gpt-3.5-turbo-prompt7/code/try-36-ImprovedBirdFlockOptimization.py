import numpy as np

class ImprovedBirdFlockOptimization(BirdFlockOptimization):
    def __init__(self, budget, dim, num_birds=20, w=0.5, c1=1.5, c2=1.5, f=0.5, cr=0.9):
        super().__init__(budget, dim, num_birds, w, c1, c2)
        self.f = f
        self.cr = cr

    def update_velocity(self, velocity, position, global_best_pos, personal_best_pos, iteration, population):
        r1, r2 = np.random.rand(), np.random.rand()
        w = self.w * (1.0 - iteration / self.budget)  # Dynamic inertia weight
        
        rand_birds = np.random.choice(self.num_birds, 3, replace=False)
        mutant = population[rand_birds[0]] + self.f * (population[rand_birds[1]] - population[rand_birds[2]])
        crossover = np.random.rand(self.dim) < self.cr
        trial = position + self.c1 * r1 * (personal_best_pos - position) + self.c2 * r2 * (global_best_pos - position)
        
        return w * velocity + np.where(crossover, trial, mutant - position)