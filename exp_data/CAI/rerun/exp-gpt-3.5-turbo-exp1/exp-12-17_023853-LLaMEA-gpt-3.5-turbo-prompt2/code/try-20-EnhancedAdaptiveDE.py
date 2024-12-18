class EnhancedAdaptiveDE(AdaptiveDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        def generate_population():
            return np.random.uniform(-5.0, 5.0, size=(self.NP, self.dim))

        population = generate_population()
        best_solution = population[np.argmin([func(ind) for ind in population])]

        for _ in range(self.budget):
            trial_population = []
            fitness_values = [func(ind) for ind in population]
            fitness_rankings = np.argsort(fitness_values)
            selection_probabilities = np.array([1 / (rank + 1) for rank in range(self.NP)]) / sum([1 / (rank + 1) for rank in range(self.NP)])

            for i in range(self.NP):
                idxs = np.random.choice(range(self.NP), 3, replace=False, p=selection_probabilities)
                a, b, c = idxs

                strategy = np.random.choice([0, 1, 2, 3, 4])

                F = self.F * np.exp(-_ / self.budget)

                if strategy == 0:
                    mutant = population[a] + F * (population[b] - population[c])
                elif strategy == 1:
                    mutant = population[a] + F * (population[b] - population[c]) + F * (population[a] - best_solution)
                elif strategy == 2:
                    mutant = best_solution + F * (population[b] - population[c])
                elif strategy == 3:
                    mutant = best_solution + F * (population[a] - best_solution) + F * (population[b] - population[c])
                else:
                    mutant = population[a] + F * (population[b] - population[c]) + F * (population[a] - best_solution) + F * (best_solution - population[c])

                j_rand = np.random.randint(self.dim)
                trial_ind = np.array([mutant[j] if np.random.rand() < self.CR or j == j_rand else population[i, j] for j in range(self.dim)])

                if func(trial_ind) < func(population[i]):
                    trial_population.append(trial_ind)
                else:
                    trial_population.append(population[i])

            population = np.array(trial_population)
            best_solution = population[np.argmin([func(ind) for ind in population])]

        return best_solution