import numpy as np

class ImprovedAdaptivePSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = int(np.ceil(8 + 2.1 * np.sqrt(self.dim)))
        self.positions = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-0.05, 0.05, (self.pop_size, self.dim))
        self.pbest_positions = np.copy(self.positions)
        self.gbest_position = np.zeros(self.dim)
        self.pbest_scores = np.full(self.pop_size, np.inf)
        self.gbest_score = np.inf
        self.c1 = 1.7  # adjusted cognitive component for better exploration
        self.c2 = 1.3  # adjusted social component for better exploitation
        self.w = 0.5   # adjusted inertia weight for stability
        self.CR = 0.9  # adjusted crossover probability for increased diversity
        self.F = 0.5   # adjusted differential weight to balance diversity
        self.eval_count = 0

    def __call__(self, func):
        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                fitness = func(self.positions[i])
                self.eval_count += 1
                
                if fitness < self.pbest_scores[i]:
                    self.pbest_scores[i] = fitness
                    self.pbest_positions[i] = self.positions[i]
                
                if fitness < self.gbest_score:
                    self.gbest_score = fitness
                    self.gbest_position = self.positions[i]
            
            if self.eval_count >= self.budget:
                break

            for i in range(self.pop_size):
                r1, r2 = np.random.rand(2)
                self.velocities[i] = (self.w * self.velocities[i] +
                                      self.c1 * r1 * (self.pbest_positions[i] - self.positions[i]) +
                                      self.c2 * r2 * (self.gbest_position - self.positions[i]))
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], -5.0, 5.0)

                # Adaptive DE Mutation and Crossover with slight modifications
                if np.random.rand() < self.CR:
                    idxs = np.random.choice(self.pop_size, 3, replace=False)
                    x0, x1, x2 = self.positions[idxs]
                    mutant_vector = x0 + self.F * (x1 - x2)
                    mutant_vector = np.clip(mutant_vector, -5.0, 5.0)
                    trial_vector = np.where(np.random.rand(self.dim) < self.CR, mutant_vector, self.positions[i])
                    
                    trial_fitness = func(trial_vector)
                    self.eval_count += 1
                    if trial_fitness < self.pbest_scores[i]:
                        self.pbest_scores[i] = trial_fitness
                        self.positions[i] = trial_vector
                        if trial_fitness < self.gbest_score:
                            self.gbest_score = trial_fitness
                            self.gbest_position = trial_vector

                    if self.eval_count >= self.budget:
                        break

        return self.gbest_position, self.gbest_score