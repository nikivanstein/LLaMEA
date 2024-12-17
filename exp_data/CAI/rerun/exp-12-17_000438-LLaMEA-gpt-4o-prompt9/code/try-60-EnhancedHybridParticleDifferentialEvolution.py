import numpy as np

class EnhancedHybridParticleDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 8 * self.dim
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.C1 = 1.5  # Cognitive component
        self.C2 = 1.5  # Social component
        self.CR = 0.9
        self.F = 0.7
        self.global_best = None
        self.global_best_fitness = float('inf')

    def __call__(self, func):
        evaluations = 0
        personal_bests = np.copy(self.population)
        personal_best_fitness = np.full(self.population_size, float('inf'))
        
        while evaluations < self.budget:
            current_fitness = np.array([func(ind) for ind in self.population])
            evaluations += self.population_size

            for i in range(self.population_size):
                if current_fitness[i] < personal_best_fitness[i]:
                    personal_best_fitness[i] = current_fitness[i]
                    personal_bests[i] = self.population[i]

                if current_fitness[i] < self.global_best_fitness:
                    self.global_best_fitness = current_fitness[i]
                    self.global_best = self.population[i]

            r1, r2 = np.random.rand(2)
            self.velocities = (self.velocities +
                               self.C1 * r1 * (personal_bests - self.population) +
                               self.C2 * r2 * (self.global_best - self.population))
            self.population = np.clip(self.population + self.velocities, self.lower_bound, self.upper_bound)

            if evaluations >= self.budget:
                break

            new_population = np.zeros_like(self.population)
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)

                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])

                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < self.global_best_fitness:
                    self.global_best_fitness = trial_fitness
                    self.global_best = trial
                if trial_fitness < func(self.population[i]):
                    new_population[i] = trial
                else:
                    new_population[i] = self.population[i]

                if evaluations >= self.budget:
                    break

            if evaluations < self.budget:
                self.population = new_population

        return self.global_best, self.global_best_fitness