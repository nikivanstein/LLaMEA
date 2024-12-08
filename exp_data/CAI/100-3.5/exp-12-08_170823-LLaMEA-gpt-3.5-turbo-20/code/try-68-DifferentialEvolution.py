class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.CR = 0.9  # Crossover rate
        self.F_min = 0.2  # Minimum value for F
        self.F_max = 0.8  # Maximum value for F

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        F_values = np.random.uniform(self.F_min, self.F_max, size=self.budget)

        for i in range(self.budget):
            for j in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]

                mutant = np.clip(a + F_values[j] * (b - c), func.bounds.lb, func.bounds.ub)

                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, population[j])

                f_trial = func(trial)
                if f_trial < fitness[j]:
                    population[j] = trial
                    fitness[j] = f_trial

                if fitness[j] < self.f_opt:
                    self.f_opt = fitness[j]
                    self.x_opt = population[j]

        return self.f_opt, self.x_opt