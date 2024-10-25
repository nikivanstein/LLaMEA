import numpy as np

class EnhancedAdaptiveDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 20
        self.f = 0.8  # DE scaling factor
        self.cr = 0.9  # DE crossover probability
        self.w = 0.7  # inertia weight for PSO
        self.c1 = 2.0  # cognitive coefficient for PSO
        self.c2 = 2.0  # social coefficient for PSO
        self.vel_max = 1.0  # maximum velocity for PSO
        self.particles = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-self.vel_max, self.vel_max, (self.population_size, self.dim))
        self.personal_best = self.particles.copy()
        self.global_best = self.particles[np.random.choice(self.population_size)]
        self.diversity_threshold = 1e-5  # diversity threshold for adaptive learning

    def __call__(self, func):
        evaluations = 0
        fitness = np.array([func(ind) for ind in self.particles])
        personal_best_fitness = fitness.copy()
        global_best_fitness = np.min(fitness)
        self.global_best = self.particles[np.argmin(fitness)]
        evaluations += self.population_size
        
        while evaluations < self.budget:
            # Apply DE operator
            for i in range(self.population_size):
                candidates = list(range(self.population_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)
                mutant = self.particles[a] + self.f * (self.particles[b] - self.particles[c])
                mutant = np.clip(mutant, self.lb, self.ub)
                trial = np.copy(self.particles[i])
                for j in range(self.dim):
                    if np.random.rand() < self.cr:
                        trial[j] = mutant[j]
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    self.particles[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < personal_best_fitness[i]:
                        personal_best_fitness[i] = trial_fitness
                        self.personal_best[i] = trial
                        if trial_fitness < global_best_fitness:
                            global_best_fitness = trial_fitness
                            self.global_best = trial

            # Check diversity for adaptive PSO
            diversity = np.mean(np.std(self.particles, axis=0))
            if diversity < self.diversity_threshold:
                self.w *= 0.95  # adapt inertia weight

            # Apply PSO operator
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.velocities[i] = (self.w * self.velocities[i] +
                                      self.c1 * r1 * (self.personal_best[i] - self.particles[i]) +
                                      self.c2 * r2 * (self.global_best - self.particles[i]))
                self.velocities[i] = np.clip(self.velocities[i], -self.vel_max, self.vel_max)
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lb, self.ub)
                current_fitness = func(self.particles[i])
                evaluations += 1
                if current_fitness < fitness[i]:
                    fitness[i] = current_fitness
                    if current_fitness < personal_best_fitness[i]:
                        personal_best_fitness[i] = current_fitness
                        self.personal_best[i] = self.particles[i]
                        if current_fitness < global_best_fitness:
                            global_best_fitness = current_fitness
                            self.global_best = self.particles[i]

        return self.global_best