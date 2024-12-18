import numpy as np

class DynamicOppositionBasedDEImproved(AdaptiveDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

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
                strategy = np.random.choice([0, 1, 2, 3, 4, 5])  # Updated mutation strategies including the dynamic adaptive opposition-based mutation

                F = self.F * np.exp(-_ / self.budget)  # Dynamic adjustment of F with budget
                CR = self.CR * np.exp(-_ / self.budget)  # Dynamic adjustment of CR with budget
                adapt_param = np.random.uniform(0.5, 1.0, size=self.dim)  # Adaptive control parameters

                if strategy == 0:
                    mutant = population[a] + F * (population[b] - population[c])
                elif strategy == 1:
                    mutant = population[a] + F * (population[b] - population[c]) + F * (population[a] - best_solution)
                elif strategy == 2:
                    mutant = best_solution + F * (population[b] - population[c])
                elif strategy == 3:
                    mutant = best_solution + F * (population[a] - best_solution) + F * (population[b] - population[c])
                elif strategy == 4:
                    mutant = population[a] + F * (population[b] - population[c]) + F * (population[a] - best_solution) + F * (best_solution - population[c])
                else:  # Updated mutation strategy for dynamic adaptive opposition-based learning with self-adaptive control parameters
                    opposite = 2 * adapt_param * best_solution - population[i]
                    mutant = adapt_param * best_solution + F * (opposite - population[a]) + F * (population[b] - adapt_param * best_solution) + F * (adapt_param * best_solution - population[c])

                j_rand = np.random.randint(self.dim)
                trial_ind = np.array([mutant[j] if np.random.rand() < CR or j == j_rand else population[i, j] for j in range(self.dim)])

                if func(trial_ind) < func(population[i]):
                    trial_population.append(trial_ind)
                else:
                    trial_population.append(population[i])

            population = np.array(trial_population)
            best_solution = population[np.argmin([func(ind) for ind in population])]

        return best_solution