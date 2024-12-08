import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.w = 0.7   # inertia weight
        self.F = 0.9   # DE scaling factor (adjusted for better exploration)
        self.CR = 0.9  # DE crossover rate
        self.velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.personal_best = self.particles.copy()
        self.personal_best_fitness = np.full(self.population_size, np.inf)
        self.global_best = None
        self.global_best_fitness = np.inf
        self.func_evals = 0

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Evaluate current particles
            fitness = np.array([func(p) for p in self.particles])
            self.func_evals += self.population_size

            # Update personal and global bests
            for i in range(self.population_size):
                if fitness[i] < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness[i]
                    self.personal_best[i] = self.particles[i]
                if fitness[i] < self.global_best_fitness:
                    self.global_best_fitness = fitness[i]
                    self.global_best = self.particles[i]

            # Hybrid Part: PSO velocity and DE mutation
            for i in range(self.population_size):
                # PSO Update
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                # Adaptive inertia weight adjustment
                self.w = 0.5 + 0.4 * (1 - self.func_evals / self.budget)  # Slightly modified time-varying inertia weight
                self.velocity[i] = (self.w * self.velocity[i] +
                                    self.c1 * r1 * (self.personal_best[i] - self.particles[i]) +
                                    self.c2 * r2 * (self.global_best - self.particles[i]))

                pso_candidate = np.clip(self.particles[i] + self.velocity[i], self.lower_bound, self.upper_bound)
                
                # DE Mutation and Crossover
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.particles[np.random.choice(idxs, 3, replace=False)]
                random_scaling = (0.5 + 0.5 * np.sin(self.func_evals * np.pi / self.budget)) * (1 + np.random.normal(0, 0.1))  # New adaptive scaling
                mutant = np.clip(a + random_scaling * self.F * (b - c) + 0.5 * self.F * (self.global_best - a) + 0.4 * (self.personal_best[i] - a) + np.random.normal(0, 0.1, self.dim), self.lower_bound, self.upper_bound) # Added weighted term

                # Dynamic DE crossover rate adjustment
                self.CR = 0.9 - (0.2 * (self.func_evals / self.budget))  # Refined adjustment for better convergence
                # Modify F for more effective DE scaling factor adjustment
                self.F = 0.8 - (0.5 * (self.func_evals / self.budget))  # Slightly more aggressive DE scaling factor adjustment
                
                cross_points = np.random.rand(self.dim) < self.CR
                de_candidate = np.where(cross_points, mutant, self.particles[i])

                # Select the better candidate
                pso_fitness = func(pso_candidate)
                de_fitness = func(de_candidate)
                self.func_evals += 2
                
                if pso_fitness < de_fitness:
                    self.particles[i] = pso_candidate
                else:
                    self.particles[i] = de_candidate

        return self.global_best