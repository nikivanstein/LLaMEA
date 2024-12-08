import numpy as np

class ImprovedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = min(80, self.budget // 6)  # Dynamic population size for initial diversity
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F_base = 0.6  # Increased for better exploration
        self.CR_base = 0.8  # Slightly reduced to maintain exploration-exploitation balance
        self.adaptation_rate = 0.1  # Higher adaptation rate for faster parameter adjustment
        self.mutation_prob = 0.6  # Adjusted mutation probability
        self.reduction_factor = 0.95  # Factor to reduce population size dynamically

    def __call__(self, func):
        # Initialize population
        population_size = self.initial_population_size
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = population_size

        # Track the best solution found
        best_index = np.argmin(fitness)
        best_individual = population[best_index]
        best_fitness = fitness[best_index]

        while eval_count < self.budget:
            for i in range(population_size):
                # Self-adaptive F and CR
                F = np.clip(self.F_base + self.adaptation_rate * np.random.randn(), 0, 1)
                CR = np.clip(self.CR_base + self.adaptation_rate * np.random.randn(), 0, 1)

                # Diverse mutation strategy choice
                if np.random.rand() < self.mutation_prob:
                    indices = np.random.choice(population_size, 4, replace=False)
                    a, b, c, d = population[indices]
                    mutant = np.clip(a + F * (b - c + d - b), self.lower_bound, self.upper_bound)
                else:
                    indices = np.random.choice(population_size, 3, replace=False)
                    a, b, c = population[indices]
                    mutant = np.clip(best_individual + F * (a - b + c - a), self.lower_bound, self.upper_bound)

                trial = np.where(np.random.rand(self.dim) < CR, mutant, population[i])

                # Evaluate trial individual
                trial_fitness = func(trial)
                eval_count += 1

                # Selection and elitism
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_individual = trial
                        best_fitness = trial_fitness

                if eval_count >= self.budget:
                    break

            # Dynamic population size adjustment
            population_size = max(int(population_size * self.reduction_factor), 4)
            population = population[:population_size]
            fitness = fitness[:population_size]

            # Reinitialize worst individuals to introduce diversity periodically
            if eval_count < self.budget // 2 and eval_count % (10 * population_size) == 0:
                worst_indices = np.argsort(fitness)[-int(0.1 * population_size):]
                for idx in worst_indices:
                    population[idx] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                    fitness[idx] = func(population[idx])
                    eval_count += 1

        # Return best found solution
        return best_individual