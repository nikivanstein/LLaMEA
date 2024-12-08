import numpy as np

class RefinedHybridDE_SA_Dynamic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 50
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.initial_population_size, self.dim))
        self.fitness = np.inf * np.ones(self.initial_population_size)
        self.best_solution = None
        self.best_fitness = np.inf
        self.dynamic_temperature = lambda evals: 1.0 - (0.98 * evals / self.budget)  # Increased cooling rate

    def evaluate_population(self, func):
        for i in range(self.population.shape[0]):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
                if self.fitness[i] < self.best_fitness:
                    self.best_fitness = self.fitness[i]
                    self.best_solution = self.population[i].copy()

    def differential_evolution(self, func, current_index, evals):
        idxs = [idx for idx in range(self.population.shape[0]) if idx != current_index]
        a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
        F = 0.6 + 0.3 * np.random.rand()  # Adaptive mutation factor
        mutant_vector = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
        crossover_rate = 0.8 + 0.1 * np.random.rand()  # Adaptive crossover rate
        crossover = np.random.rand(self.dim) < crossover_rate
        trial_vector = np.where(crossover, mutant_vector, self.population[current_index])
        
        trial_fitness = func(trial_vector)
        if trial_fitness < self.fitness[current_index]:
            self.population[current_index] = trial_vector
            self.fitness[current_index] = trial_fitness

        if trial_fitness < self.best_fitness:
            self.best_fitness = trial_fitness
            self.best_solution = trial_vector.copy()

    def simulated_annealing(self, func, index, evals):
        current_temp = self.dynamic_temperature(evals)
        idxs = np.random.choice(range(self.dim), (self.dim + 1) // 3, replace=False)  # More dimensions perturbed
        trial = self.population[index].copy()
        trial[idxs] += np.random.normal(0, 0.5, size=idxs.shape)  # Gaussian perturbation
        trial = np.clip(trial, self.lower_bound, self.upper_bound)
        
        trial_fitness = func(trial)
        current_fitness = self.fitness[index]

        if trial_fitness < current_fitness or np.random.rand() < np.exp((current_fitness - trial_fitness) / current_temp):
            self.population[index] = trial
            self.fitness[index] = trial_fitness
            if trial_fitness < self.best_fitness:
                self.best_fitness = trial_fitness
                self.best_solution = trial.copy()

    def resize_population(self, evals):
        if evals > self.budget * 0.4 and self.population.shape[0] > 10:  # Earlier reduction
            reduced_size = max(10, int(self.initial_population_size * (1 - evals / self.budget)))
            self.population = self.population[:reduced_size]
            self.fitness = self.fitness[:reduced_size]

    def __call__(self, func):
        evals = 0
        self.evaluate_population(func)
        evals += self.population.shape[0]

        while evals < self.budget:
            for i in range(self.population.shape[0]):
                if evals >= self.budget:
                    break
                self.differential_evolution(func, i, evals)
                self.simulated_annealing(func, i, evals)
                evals += 1
            self.resize_population(evals)

        return self.best_solution, self.best_fitness