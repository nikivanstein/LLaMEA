import numpy as np

class EnhancedDE(DifferentialEvolution):
    def __init__(self, budget, dim, Cr=0.9, F=0.8, pop_size=50, F_lb=0.2, F_ub=0.9, F_adapt=0.1):
        super().__init__(budget, dim, Cr, F, pop_size)
        self.F_lb = F_lb
        self.F_ub = F_ub
        self.F_adapt = F_adapt

    def __call__(self, func):
        def adapt_mutation_factor(F):
            return np.clip(F + np.random.uniform(-self.F_adapt, self.F_adapt), self.F_lb, self.F_ub)

        def create_population():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))

        def clip_to_bounds(x):
            return np.clip(x, self.lower_bound, self.upper_bound)

        def calculate_crowding_distance(fitness_values):
            crowding_distance = np.zeros(self.pop_size)
            sorted_indices = np.argsort(fitness_values)
            crowding_distance[sorted_indices[[0, -1]]] = np.inf
            norm_fitness = (fitness_values - np.min(fitness_values)) / (np.max(fitness_values) - np.min(fitness_values))
            for i in range(1, self.pop_size - 1):
                crowding_distance[sorted_indices[i]] += norm_fitness[sorted_indices[i+1]] - norm_fitness[sorted_indices[i-1]]
            return crowding_distance

        population = create_population()
        fitness_values = np.array([func(ind) for ind in population])
        evals = self.pop_size

        while evals < self.budget:
            new_population = []
            crowding_distance = calculate_crowding_distance(fitness_values)
            
            for i in range(self.pop_size):
                a, b, c = np.random.choice(self.pop_size, 3, replace=False)
                mutant = clip_to_bounds(population[a] + self.F * (population[b] - population[c]))
                crossover = np.random.rand(self.dim) < self.Cr
                trial = population[i].copy()
                trial[crossover] = mutant[crossover]
                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < fitness_values[i]:
                    population[i] = trial
                    fitness_values[i] = trial_fitness

            best_idx = np.argmin(fitness_values)
            best_solution = population[best_idx]
            self.F = adapt_mutation_factor(self.F)

        return best_solution