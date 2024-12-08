import numpy as np

class EnhancedAdaptiveLévyDifferentialEvolutionV3:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(4 * dim, 20)
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.eval_count = 0
        self.rejuvenation_interval = self.budget // 5  # Periodic rejuvenation every 20% of the budget

    def levy_flight(self, beta=1.5):
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / np.abs(v)**(1 / beta)
        return step

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
                self.eval_count += 1

    def mutate(self, target_idx):
        indices = np.delete(np.arange(self.population_size), target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        F1 = 0.6 + 0.4 * np.random.rand()  # Slightly higher dynamic scaling factor
        mutant1 = self.population[a] + F1 * (self.population[b] - self.population[c])
        
        d, e, f = np.random.choice(indices, 3, replace=False)
        F2 = 0.7 + 0.2 * np.random.rand()  # Alternative strategy with adjusted scaling
        mutant2 = self.population[d] + F2 * (self.population[e] - self.population[f])

        return mutant1 if np.random.rand() < 0.6 else mutant2  # Increased preference for mutant1

    def crossover(self, target, mutant):
        CR = 0.95 * (1 - self.eval_count / self.budget)  # Higher adaptive crossover probability
        crossover = np.random.rand(self.dim) < CR
        return np.where(crossover, mutant, target)

    def select(self, target_idx, trial, func):
        trial_fitness = func(trial)
        self.eval_count += 1
        if trial_fitness < self.fitness[target_idx]:
            self.population[target_idx] = trial
            self.fitness[target_idx] = trial_fitness

    def rejuvenate_population(self):
        rejuvenation_factor = 0.1
        for i in range(self.population_size):
            if np.random.rand() < rejuvenation_factor:
                self.population[i] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                self.fitness[i] = np.inf

    def __call__(self, func):
        self.evaluate_population(func)

        while self.eval_count < self.budget:
            if self.eval_count % self.rejuvenation_interval == 0:
                self.rejuvenate_population()
                self.evaluate_population(func)

            for i in range(self.population_size):
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)
                trial = np.clip(trial, self.lower_bound, self.upper_bound)

                if np.random.rand() < 0.35:  # Further increased probability for applying Lévy flight
                    trial += 0.005 * self.levy_flight()  # Slightly adjusted step size

                self.select(i, trial, func)

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]