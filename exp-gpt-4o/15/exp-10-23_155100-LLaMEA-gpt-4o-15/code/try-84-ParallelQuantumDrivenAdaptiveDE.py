import numpy as np
from concurrent.futures import ThreadPoolExecutor

class ParallelQuantumDrivenAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 8 * dim
        self.population_size = self.initial_population_size
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.f_min = 0.5
        self.f_max = 0.9
        self.cr_min = 0.4
        self.cr_max = 0.9
        self.elite_count = max(2, self.population_size // 8)
        self.entropy_threshold = 0.02
        self.chaos_control = 1.8
        self.exploration_probability = 0.15

    def adaptive_parameters(self):
        f = np.random.uniform(self.f_min, self.f_max)
        cr = np.random.uniform(self.cr_min, self.cr_max)
        return f, cr

    def gradient_based_perturbation(self, sol, func):
        if np.random.rand() < self.exploration_probability:
            grad = np.random.randn(self.dim)
            step_size = 0.05
            return np.clip(sol - step_size * grad, self.lower_bound, self.upper_bound)
        return sol

    def chaotic_map(self, x):
        return (self.chaos_control * x + 0.7) % 1.0

    def mutate(self, target_idx, f):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        chaos_factor = self.chaotic_map(np.random.rand())
        mutant = np.clip(a + f * (b - c) * chaos_factor, self.lower_bound, self.upper_bound)
        return mutant

    def crossover(self, target, mutant, cr):
        crossover = np.random.rand(self.dim) < cr
        if not np.any(crossover):
            crossover[np.random.randint(0, self.dim)] = True
        return np.where(crossover, mutant, target)

    def calculate_entropy(self):
        return np.mean(np.std(self.population, axis=0))

    def evaluate_individual(self, i, fitness, func, eval_count):
        f, cr = self.adaptive_parameters()
        target = self.population[i]
        mutant = self.mutate(i, f)
        trial = self.crossover(target, mutant, cr)

        if self.calculate_entropy() < self.entropy_threshold:
            trial = self.gradient_based_perturbation(trial, func)

        trial_fitness = func(trial)
        eval_count += 1
        if trial_fitness < self.best_fitness:
            self.best_solution = trial
            self.best_fitness = trial_fitness

        if trial_fitness < func(target):
            self.population[i] = trial
            fitness[i] = trial_fitness

        return eval_count

    def __call__(self, func):
        eval_count = 0
        fitness = np.full(self.population_size, np.inf)

        while eval_count < self.budget:
            dynamic_population_size = max(6, self.initial_population_size * (1 - eval_count / self.budget))
            self.population_size = int(dynamic_population_size)
            self.population = self.population[:self.population_size]
            fitness = fitness[:self.population_size]

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(self.evaluate_individual, i, fitness, func, eval_count) for i in range(self.population_size)]
                
                for future in futures:
                    eval_count += future.result()

                if eval_count >= self.budget:
                    break

            elite_indices = np.argsort(fitness)[:self.elite_count]
            elites = self.population[elite_indices]
            self.population[:self.elite_count] = elites

        return self.best_solution, self.best_fitness