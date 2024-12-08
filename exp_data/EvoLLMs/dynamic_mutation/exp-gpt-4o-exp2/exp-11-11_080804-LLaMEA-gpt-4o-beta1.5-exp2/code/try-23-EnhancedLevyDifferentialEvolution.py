import numpy as np

class EnhancedLevyDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = max(10, 5 * dim)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.f = 0.5
        self.cr = 0.9
        self.alpha = 1.5

    def levy_flight(self, size):
        u = np.random.normal(0, 1, size) * (np.sqrt(np.abs(np.random.normal(0, 1, size))) ** (-1 / self.alpha))
        return u

    def adaptive_mutation_scaling(self, evaluations):
        return self.f * (1 - (evaluations / self.budget))

    def dynamic_crossover_probability(self, evaluations):
        return self.cr * (0.5 + 0.5 * (self.budget - evaluations) / self.budget)

    def adjust_population_size(self, evaluations):
        # Dynamically adjust population size based on remaining budget
        return max(10, int(self.initial_population_size * (1 - evaluations / self.budget)))

    def __call__(self, func):
        population_size = self.initial_population_size
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = population_size
        
        while evaluations < self.budget:
            population_size = self.adjust_population_size(evaluations)
            new_population = np.copy(population[:population_size])
            
            for i in range(population_size):
                # Mutation with adaptive scaling and noise
                indices = np.random.choice(population_size, 3, replace=False)
                x0, x1, x2 = new_population[indices]
                adaptive_f = self.adaptive_mutation_scaling(evaluations)
                noise = np.random.normal(0, 0.1, self.dim)
                mutant = np.clip(x0 + adaptive_f * (x1 - x2 + noise), self.lower_bound, self.upper_bound)
                
                # Dynamic Crossover with permutation
                crossover_prob = self.dynamic_crossover_probability(evaluations)
                crossover = np.random.rand(self.dim) < crossover_prob
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                perm = np.random.permutation(self.dim)
                trial = np.where(crossover, mutant[perm], new_population[i])
                
                # Enhanced Levy flight with adaptive step size
                step_size = (self.budget - evaluations) / self.budget
                levy_step = self.levy_flight(self.dim)
                trial += step_size * levy_step * np.exp(-evaluations / self.budget)
                trial = np.clip(trial, self.lower_bound, self.upper_bound)
                
                # Selection with elitism
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness

                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best = trial

                if evaluations >= self.budget:
                    break

            population = np.copy(new_population)

        return best