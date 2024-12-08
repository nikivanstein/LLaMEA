import numpy as np

class QuantumEnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 12 * dim  # Increased population size for better diversity
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.f_min = 0.4  # Adjusted min differential weight
        self.f_max = 0.9  # Max differential weight
        self.cr_min = 0.6  # Adjusted min crossover probability
        self.cr_max = 0.95  # Max crossover probability
        self.elite_count = max(1, self.population_size // 10)  # Elite preservation

    def adaptive_parameters(self):
        f = np.random.uniform(self.f_min, self.f_max)
        cr = np.random.uniform(self.cr_min, self.cr_max)
        return f, cr

    def stochastic_phase_adjustment(self, sol):
        theta = np.random.uniform(0, 2 * np.pi, self.dim)
        perturbation = (np.random.rand() - 0.5) * (self.upper_bound - self.lower_bound) / 20  # Adjusted perturbation scale
        return sol + np.sin(theta) * perturbation

    def mutate(self, target_idx, f):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        mutant = np.clip(a + f * (b - c), self.lower_bound, self.upper_bound)
        return mutant

    def crossover(self, target, mutant, cr):
        crossover = np.random.rand(self.dim) < cr
        if not np.any(crossover):
            crossover[np.random.randint(0, self.dim)] = True
        return np.where(crossover, mutant, target)

    def __call__(self, func):
        eval_count = 0
        fitness = np.full(self.population_size, np.inf)

        while eval_count < self.budget:
            for i in range(self.population_size):
                f, cr = self.adaptive_parameters()
                target = self.population[i]
                mutant = self.mutate(i, f)
                trial = self.crossover(target, mutant, cr)
                trial = self.stochastic_phase_adjustment(trial)

                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < self.best_fitness:
                    self.best_solution = trial
                    self.best_fitness = trial_fitness

                if trial_fitness < func(target):
                    self.population[i] = trial
                    fitness[i] = trial_fitness

                if eval_count >= self.budget:
                    break

            elite_indices = np.argsort(fitness)[:self.elite_count]
            elites = self.population[elite_indices]
            self.population[:self.elite_count] = elites

        return self.best_solution, self.best_fitness