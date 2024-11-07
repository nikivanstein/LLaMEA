import numpy as np

class DynamicEnsembleChaoticPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        
        self.population_size = 60  # Adjusted population size for dynamic ensemble
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.population_size, self.dim)) * 0.3  # Slightly reduced velocity scaling
        
        self.pbest_positions = np.copy(self.particles)
        self.pbest_scores = np.full(self.population_size, np.inf)
        
        self.gbest_position = None
        self.gbest_score = np.inf
        
        self.c1, self.c2 = 2.0, 2.0  # Fixed cognitive and social parameters for simplicity
        self.w_initial, self.w_final = 0.9, 0.4    # Broadened inertia weight range
        self.scale_factor_initial = 0.6
        self.scale_factor_final = 0.9

    def chaotic_mapping(self, x):
        return 4 * x * (1 - x)  # Logistic map

    def __call__(self, func):
        evaluations = 0
        chaotic_factor = np.random.rand()

        while evaluations < self.budget:
            progress = evaluations / self.budget
            self.w = self.w_initial - progress * (self.w_initial - self.w_final)
            self.scale_factor = self.scale_factor_initial + progress * (self.scale_factor_final - self.scale_factor_initial)
            chaotic_factor = self.chaotic_mapping(chaotic_factor)

            for i in range(self.population_size):
                score = func(self.particles[i])
                evaluations += 1

                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_positions[i] = self.particles[i]

                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest_position = self.particles[i]

                if evaluations >= self.budget:
                    break
            
            if evaluations < 0.3 * self.budget:  # Use PSO in the initial phase
                for i in range(self.population_size):
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    inertia = chaotic_factor
                    cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.particles[i])
                    social = self.c2 * r2 * (self.gbest_position - self.particles[i])
                    self.velocities[i] = inertia * self.w * self.velocities[i] + cognitive + social
                    self.particles[i] += self.velocities[i]
                    self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)
            else:  # Switch to DE in the latter phase
                for i in range(self.population_size):
                    indices = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    mutant = self.particles[a] + self.scale_factor * (self.particles[b] - self.particles[c])
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                    crossover_rate = 0.8
                    crossover_mask = np.random.rand(self.dim) < crossover_rate
                    trial = np.where(crossover_mask, mutant, self.particles[i])
                    
                    trial_score = func(trial)
                    evaluations += 1

                    if trial_score < self.pbest_scores[i]:
                        self.pbest_scores[i] = trial_score
                        self.pbest_positions[i] = trial

                    if trial_score < self.gbest_score:
                        self.gbest_score = trial_score
                        self.gbest_position = trial

                    if evaluations >= self.budget:
                        break

            # Adaptive re-initialization strategy
            if evaluations / self.budget > 0.7 and np.all(np.abs(self.velocities) < 0.01):
                self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
                self.velocities = np.random.uniform(-1.0, 1.0, (self.population_size, self.dim)) * 0.3

        return self.gbest_score, self.gbest_position