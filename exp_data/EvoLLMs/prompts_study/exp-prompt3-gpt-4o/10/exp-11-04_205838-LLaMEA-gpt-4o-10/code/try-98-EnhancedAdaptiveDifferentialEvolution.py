import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.max_trials_without_improvement = 100
        self.f_initial = 0.8
        self.cr_initial = 0.9
        # Use chaotic maps for initial population generation
        self.population = self.init_population_with_chaos(self.initial_population_size, dim)
        self.best_solution = None
        self.best_fitness = np.inf
        self.eval_count = 0

    def init_population_with_chaos(self, pop_size, dim):
        """Generate initial population using chaotic maps."""
        chaotic_map = np.sin(np.arange(pop_size * dim) * 4.0 * np.pi / (pop_size * dim))
        chaotic_map = (chaotic_map + 1) / 2  # Normalize to [0, 1]
        scaling_factor = 0.5 + 0.5 * np.std(chaotic_map)  # Adaptive scaling based on diversity
        chaotic_pop = self.lower_bound + scaling_factor * chaotic_map * (self.upper_bound - self.lower_bound)
        return chaotic_pop.reshape((pop_size, dim))

    def calculate_diversity(self):
        """Calculate population diversity to adapt mutation factor."""
        return np.mean(np.std(self.population, axis=0))

    def levy_flight(self, L=1.5):
        """Generate Levy flight steps."""
        u = np.random.normal(0, 1, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / (np.fabs(v) ** (1.0 / L))
        return step

    def __call__(self, func):
        fitness = np.apply_along_axis(func, 1, self.population)
        self.eval_count += self.population_size

        best_idx = np.argmin(fitness)
        self.best_solution = self.population[best_idx]
        self.best_fitness = fitness[best_idx]

        trials_without_improvement = 0

        while self.eval_count < self.budget:
            diversity = self.calculate_diversity()
            progress_ratio = self.eval_count / self.budget
            self.f = self.f_initial * (1 + 0.2 * (1 - diversity) * progress_ratio)

            self.population_size = max(5, int(self.initial_population_size * (1 - progress_ratio)))

            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break

                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = self.population[a] + self.f * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                self.cr = self.cr_initial * (1 - 0.5 * progress_ratio) + 0.5 * np.random.rand()
                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])

                trial_fitness = func(trial)
                self.eval_count += 1

                if trial_fitness < fitness[i]:
                    self.population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < self.best_fitness:
                        self.best_solution = trial
                        self.best_fitness = trial_fitness
                        trials_without_improvement = 0
                else:
                    trials_without_improvement += 1

                if trials_without_improvement >= self.max_trials_without_improvement:
                    perturbation = self.levy_flight() * (0.5 + 0.5 * np.random.rand())  # Adaptive scaling
                    local_solution = self.best_solution + perturbation
                    local_solution = np.clip(local_solution, self.lower_bound, self.upper_bound)
                    local_fitness = func(local_solution)
                    self.eval_count += 1
                    if local_fitness < self.best_fitness:
                        self.best_solution = local_solution
                        self.best_fitness = local_fitness
                    self.f_initial = np.random.uniform(0.5, 1)
                    self.cr_initial = np.random.uniform(0.7, 1)
                    trials_without_improvement = 0

        return self.best_solution