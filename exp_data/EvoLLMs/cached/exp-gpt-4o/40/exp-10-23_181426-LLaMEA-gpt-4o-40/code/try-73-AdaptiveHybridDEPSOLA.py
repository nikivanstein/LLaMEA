import numpy as np

class AdaptiveHybridDEPSOLA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.initial_de_f = 0.8
        self.initial_de_cr = 0.85
        self.initial_pso_w = 0.5
        self.pso_c1 = 1.0
        self.pso_c2 = 1.2
        self.lb = -5.0
        self.ub = 5.0
        self.max_evaluations = budget
        self.eval_count = 0
        self.learning_rate = 0.1
        self.strategies_prob = np.array([0.5, 0.5])  # DE, PSO

    def __call__(self, func):
        np.random.seed(42)
        self.eval_count = 0
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocity = np.zeros((self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.eval_count += self.population_size
        personal_best = population.copy()
        personal_best_fitness = fitness.copy()
        global_best_idx = np.argmin(fitness)
        global_best = population[global_best_idx]

        while self.eval_count < self.max_evaluations:
            progress_ratio = self.eval_count / self.max_evaluations
            de_f = self.initial_de_f * (1 - progress_ratio) + 0.5 * progress_ratio
            de_cr = self.initial_de_cr * (1 - progress_ratio) + 0.6 * progress_ratio
            pso_w = self.initial_pso_w * (1 - progress_ratio) + 0.4 * progress_ratio

            strategy_choice = np.random.choice([0, 1], p=self.strategies_prob)
            
            for i in range(self.population_size):
                if self.eval_count >= self.max_evaluations:
                    break

                if strategy_choice == 0:  # Differential Evolution
                    idxs = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                    mutant = np.clip(a + de_f * (b - c), self.lb, self.ub)
                    cross_points = np.random.rand(self.dim) < de_cr
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    trial = np.where(cross_points, mutant, population[i])
                    trial_fitness = func(trial)
                    self.eval_count += 1
                    if trial_fitness < fitness[i]:
                        population[i] = trial
                        fitness[i] = trial_fitness
                        if trial_fitness < personal_best_fitness[i]:
                            personal_best[i] = trial
                            personal_best_fitness[i] = trial_fitness
                            if trial_fitness < personal_best_fitness[global_best_idx]:
                                global_best = trial
                                global_best_idx = i

                else:  # Particle Swarm Optimization
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    velocity[i] = (pso_w * velocity[i] +
                                   self.pso_c1 * r1 * (personal_best[i] - population[i]) +
                                   self.pso_c2 * r2 * (global_best - population[i]))
                    population[i] = np.clip(population[i] + velocity[i], self.lb, self.ub)
                    fitness[i] = func(population[i])
                    self.eval_count += 1
                    if fitness[i] < personal_best_fitness[i]:
                        personal_best[i] = population[i]
                        personal_best_fitness[i] = fitness[i]
                        if fitness[i] < personal_best_fitness[global_best_idx]:
                            global_best = population[i]
                            global_best_idx = i

            # Encourage diversity
            if self.eval_count >= self.max_evaluations * 0.70:
                diversity_count = max(1, int(self.population_size * 0.2))
                for _ in range(diversity_count):
                    if self.eval_count < self.max_evaluations:
                        idx = np.random.randint(0, self.population_size)
                        population[idx] = np.random.uniform(self.lb, self.ub, self.dim)
                        fitness[idx] = func(population[idx])
                        self.eval_count += 1
                        if fitness[idx] < personal_best_fitness[idx]:
                            personal_best[idx] = population[idx]
                            personal_best_fitness[idx] = fitness[idx]
                            if fitness[idx] < personal_best_fitness[global_best_idx]:
                                global_best = population[idx]
                                global_best_idx = idx

            # Update strategy probabilities using Learning Automata
            best_strategy = np.argmin([np.min(fitness), np.min(fitness)])
            self.strategies_prob = np.clip(self.strategies_prob + self.learning_rate * (1 - self.strategies_prob[best_strategy]), 0, 1)
            self.strategies_prob /= np.sum(self.strategies_prob)

        return global_best, personal_best_fitness[global_best_idx]