import numpy as np

class AdaptiveParticleSwarmFirefly:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = int(10 + 2 * np.sqrt(dim))
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.func_evaluations = 0
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_score = np.inf
        self.global_best_position = None
        self.alpha = 0.5  # Firefly attractiveness scaling factor
        self.beta = 1.0   # Base attractiveness
        self.gamma = 1.0  # Absorption coefficient
        self.inertia_weight = 0.9  # Particle inertia

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            for i in range(self.population_size):
                # Evaluate fitness
                score = func(self.population[i])
                self.func_evaluations += 1
                
                # Update personal best
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.population[i]
                
                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.population[i]
            
            # Update velocities and positions using firefly inspired dynamics
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if self.personal_best_scores[j] < self.personal_best_scores[i]:
                        distance = np.linalg.norm(self.population[i] - self.population[j])
                        attractiveness = self.beta * np.exp(-self.gamma * distance**2)
                        self.velocities[i] += self.alpha * (self.population[j] - self.population[i]) * attractiveness
            
            # Particle swarm-like update with adaptive inertia weight
            self.inertia_weight = 0.4 + 0.5 * (1 - self.func_evaluations / self.budget)
            self.velocities = self.inertia_weight * self.velocities + np.random.uniform(-1, 1, (self.population_size, self.dim))
            self.population += self.velocities
            self.population = np.clip(self.population, self.lower_bound, self.upper_bound)

        return self.global_best_position