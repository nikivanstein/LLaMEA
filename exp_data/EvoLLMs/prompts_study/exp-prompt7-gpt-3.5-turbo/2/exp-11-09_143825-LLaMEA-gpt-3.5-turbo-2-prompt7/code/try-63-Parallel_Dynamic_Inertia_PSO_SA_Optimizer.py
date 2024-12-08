import numpy as np
from concurrent.futures import ThreadPoolExecutor

class Parallel_Dynamic_Inertia_PSO_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.particle_pos = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.particle_vel = np.random.uniform(-1, 1, (budget, dim))
        self.global_best = np.random.uniform(-5.0, 5.0, dim)
        self.global_best_score = float('inf')
        self.temperature = 1.0
        self.inertia_weight = 0.7

    def __call__(self, func):
        with ThreadPoolExecutor() as executor:
            for _ in range(self.budget):
                futures = [executor.submit(func, self.particle_pos[i]) for i in range(self.budget)]
                results = [future.result() for future in futures]
                
                for i, fitness in enumerate(results):
                    if fitness < self.global_best_score:
                        self.global_best_score = fitness
                        self.global_best = self.particle_pos[i]
                    if fitness < func(self.global_best):
                        self.global_best = self.particle_pos[i]

                    dynamic_inertia_weight = np.exp(-0.1 * _ / self.budget)  # Dynamic adaptation of inertia weight
                    mutation_rate = 1.5  # Increased mutation rate for exploration
                    new_vel = dynamic_inertia_weight * self.particle_vel[i] + mutation_rate * np.random.uniform(0, 1) * (self.global_best - self.particle_pos[i]) + mutation_rate * np.random.uniform(0, 1) * (self.particle_pos[i] - self.particle_pos[i])
                    new_pos = self.particle_pos[i] + new_vel
                    new_pos = np.clip(new_pos, -5.0, 5.0)
                    new_fitness = results[i]

                    if new_fitness < fitness or np.random.rand() < np.exp((fitness - new_fitness) / self.temperature):
                        self.particle_pos[i] = new_pos
                        self.particle_vel[i] = new_vel

                self.temperature *= 0.95  # Annealing schedule

        return self.global_best