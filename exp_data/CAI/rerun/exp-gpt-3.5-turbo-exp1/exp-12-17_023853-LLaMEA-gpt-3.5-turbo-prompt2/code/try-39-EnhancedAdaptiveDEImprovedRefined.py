class EnhancedAdaptiveDEImprovedRefined(EnhancedAdaptiveDEImproved):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        def generate_population():
            return np.random.uniform(-5.0, 5.0, size=(self.NP, self.dim))

        population = generate_population()
        best_solution = population[np.argmin([func(ind) for ind in population])]
        convergence_rate = 0.0  # Initializing convergence rate
        prev_best_solution = best_solution
        
        for _ in range(self.budget):
            trial_population = []
            for i in range(self.NP):
                idxs = [idx for idx in range(self.NP) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                strategy = np.random.choice([0, 1, 2, 3, 4])  # Added new mutation strategy

                F = self.F * np.exp(-_ / self.budget)  # Dynamic adjustment of F
                CR = self.CR * np.exp(-_ / self.budget)  # Dynamic adjustment of CR
                
                if strategy == 0:
                    mutant = population[a] + F * (population[b] - population[c])
                elif strategy == 1:
                    mutant = population[a] + F * (population[b] - population[c]) + F * (population[a] - best_solution)
                elif strategy == 2:
                    mutant = best_solution + F * (population[b] - population[c])
                elif strategy == 3:
                    mutant = best_solution + F * (population[a] - best_solution) + F * (population[b] - population[c])
                else:  # New mutation strategy
                    mutant = population[a] + F * (population[b] - population[c]) + F * (population[a] - best_solution) + F * (best_solution - population[c])
                
                j_rand = np.random.randint(self.dim)
                trial_ind = np.array([mutant[j] if np.random.rand() < CR or j == j_rand else population[i, j] for j in range(self.dim)])

                if func(trial_ind) < func(population[i]):
                    trial_population.append(trial_ind)
                else:
                    trial_population.append(population[i])

            population = np.array(trial_population)
            best_solution = population[np.argmin([func(ind) for ind in population])]
            if func(best_solution) == func(prev_best_solution):  # Check convergence
                convergence_rate += 1.0

            if convergence_rate >= 0.1:  # Adapt population size dynamically
                self.NP = int(self.NP * 1.1)
                population = generate_population()
                convergence_rate = 0.0

            prev_best_solution = best_solution

        return best_solution