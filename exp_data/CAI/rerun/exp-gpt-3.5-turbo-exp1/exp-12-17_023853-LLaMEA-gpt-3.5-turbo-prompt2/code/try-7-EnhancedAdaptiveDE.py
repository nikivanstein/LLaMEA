class EnhancedAdaptiveDE(AdaptiveDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.success_rate = 0.5  # Initialize success rate

    def __call__(self, func):
        def generate_population():
            return np.random.uniform(-5.0, 5.0, size=(self.NP, self.dim))

        population = generate_population()
        best_solution = population[np.argmin([func(ind) for ind in population])]

        for _ in range(self.budget):
            trial_population = []
            for i in range(self.NP):
                idxs = [idx for idx in range(self.NP) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                strategy = np.random.choice([0, 1, 2, 3, 4])  # Added new mutation strategy

                if strategy == 0:
                    F_mut = self.F  # Original DE mutation factor
                else:
                    F_mut = self.F * np.random.uniform(0.5, 1.0) + 0.5 * np.random.uniform(0, 0.5)  # Adaptive mutation factor

                if strategy == 0:
                    mutant = population[a] + F_mut * (population[b] - population[c])
                elif strategy == 1:
                    mutant = population[a] + F_mut * (population[b] - population[c]) + F_mut * (population[a] - best_solution)
                elif strategy == 2:
                    mutant = best_solution + F_mut * (population[b] - population[c])
                elif strategy == 3:
                    mutant = best_solution + F_mut * (population[a] - best_solution) + F_mut * (population[b] - population[c])
                else:  # New mutation strategy
                    mutant = population[a] + F_mut * (population[b] - population[c]) + F_mut * (population[a] - best_solution) + F_mut * (best_solution - population[c])

                j_rand = np.random.randint(self.dim)
                trial_ind = np.array([mutant[j] if np.random.rand() < self.CR or j == j_rand else population[i, j] for j in range(self.dim)])

                if func(trial_ind) < func(population[i]):
                    trial_population.append(trial_ind)
                    self.success_rate = 0.2 * 1 + 0.8 * self.success_rate  # Update success rate
                else:
                    trial_population.append(population[i])
                    self.success_rate = 0.2 * 0 + 0.8 * self.success_rate  # Update success rate

            population = np.array(trial_population)
            best_solution = population[np.argmin([func(ind) for ind in population])]

        return best_solution