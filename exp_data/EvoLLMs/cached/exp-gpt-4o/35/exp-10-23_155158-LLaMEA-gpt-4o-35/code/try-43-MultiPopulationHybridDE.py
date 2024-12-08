import numpy as np

class MultiPopulationHybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_populations = 3
        self.population_size = 5 * dim
        self.mutation_factor_base = 0.7
        self.crossover_prob_base = 0.9
        self.rescale_interval = max(100, budget // 10)
        self.success_rate_history = [[] for _ in range(self.num_populations)]
        
    def adapt_parameters(self, population_id):
        if len(self.success_rate_history[population_id]) >= 5:
            recent_success = np.mean(self.success_rate_history[population_id][-5:])
            mutation_factor = 0.6 + 0.3 * recent_success
            crossover_prob = 0.65 + 0.25 * recent_success
            return mutation_factor, crossover_prob
        return self.mutation_factor_base, self.crossover_prob_base

    def intensification(self, best_individual):
        step_size = 0.05 + 0.03 * np.random.rand() * np.random.choice([1, -1])
        perturbation = np.random.uniform(-step_size, step_size, self.dim)
        candidate = np.clip(best_individual + perturbation, self.lower_bound, self.upper_bound)
        return candidate

    def __call__(self, func):
        populations = [np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
                       for _ in range(self.num_populations)]
        fitnesses = [np.array([func(ind) for ind in pop]) for pop in populations]
        budget_used = self.population_size * self.num_populations
        generation = 0

        while budget_used < self.budget:
            for pop_id in range(self.num_populations):
                mutation_factor, crossover_prob = self.adapt_parameters(pop_id)
                pop, fit = populations[pop_id], fitnesses[pop_id]
                new_population = []
                new_fitness = []

                for i in range(self.population_size):
                    idxs = np.delete(np.arange(self.population_size), i)
                    a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                    mutant_vector = np.clip(a + mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                    cross_points = np.random.rand(self.dim) < crossover_prob
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    trial_vector = np.where(cross_points, mutant_vector, pop[i])
                    trial_fitness = func(trial_vector)
                    budget_used += 1

                    if trial_fitness < fit[i]:
                        new_population.append(trial_vector)
                        new_fitness.append(trial_fitness)
                        self.success_rate_history[pop_id].append(1)
                    else:
                        new_population.append(pop[i])
                        new_fitness.append(fit[i])
                        self.success_rate_history[pop_id].append(0)

                    if budget_used >= self.budget:
                        break

                populations[pop_id] = np.array(new_population)
                fitnesses[pop_id] = np.array(new_fitness)

                best_idx = np.argmin(fitnesses[pop_id])
                if budget_used < self.budget and np.random.rand() < 0.4:
                    intense_candidate = self.intensification(populations[pop_id][best_idx])
                    intense_fitness = func(intense_candidate)
                    budget_used += 1

                    if intense_fitness < fitnesses[pop_id][best_idx]:
                        populations[pop_id][best_idx] = intense_candidate
                        fitnesses[pop_id][best_idx] = intense_fitness

            generation += 1

            if budget_used >= self.budget:
                break

        best_overall_idx = np.argmin([np.min(fit) for fit in fitnesses])
        best_pop_idx = np.argmin(fitnesses[best_overall_idx])
        return populations[best_overall_idx][best_pop_idx], fitnesses[best_overall_idx][best_pop_idx]