import numpy as np

class AdaptiveFractalDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 20
        self.population_growth_rate = 0.1  # Allows for dynamic population resizing
        self.f = 0.8
        self.cr = 0.9
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        self.vel_max = 1.0
        self.particles = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-self.vel_max, self.vel_max, (self.population_size, self.dim))
        self.personal_best = self.particles.copy()
        self.global_best = self.particles[np.random.choice(self.population_size)]
        self.diversity_threshold = 1e-5
        self.diversity_probability = 0.05
        self.iteration = 0

    def fractal_map(self, x):
        return (np.sin(x * np.pi) + 1) / 2  # Fractal-inspired transformation

    def __call__(self, func):
        evaluations = 0
        fitness = np.array([func(ind) for ind in self.particles])
        personal_best_fitness = fitness.copy()
        global_best_fitness = np.min(fitness)
        self.global_best = self.particles[np.argmin(fitness)]
        evaluations += self.population_size
        
        while evaluations < self.budget:
            self.iteration += 1

            # Apply DE operator with fractal map influence
            for i in range(self.population_size):
                candidates = list(range(self.population_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)
                fractal_factor = self.fractal_map(self.iteration / 100.0)
                mutant = self.particles[a] + fractal_factor * self.f * (self.particles[b] - self.particles[c])
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

            # Dynamic population resizing
            if np.random.rand() < self.population_growth_rate:
                new_individuals = np.random.uniform(self.lb, self.ub, (2, self.dim))
                new_fitness = np.array([func(ind) for ind in new_individuals])
                evaluations += 2
                self.particles = np.vstack((self.particles, new_individuals))
                fitness = np.append(fitness, new_fitness)
                personal_best_fitness = np.append(personal_best_fitness, new_fitness)
                self.personal_best = np.vstack((self.personal_best, new_individuals))
                self.population_size += 2

            # Adjust inertia weight based on diversity
            diversity = np.mean(np.std(self.particles, axis=0))
            if diversity < self.diversity_threshold:
                self.w *= 0.9

            # Dynamic parameter tuning for PSO
            self.c1 = 1.5 + 0.5 * np.cos(self.iteration / 50.0)
            self.c2 = 1.5 + 0.5 * np.sin(self.iteration / 50.0)

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