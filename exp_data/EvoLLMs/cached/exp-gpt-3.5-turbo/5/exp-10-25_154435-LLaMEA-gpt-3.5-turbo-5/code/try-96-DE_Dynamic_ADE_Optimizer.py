import numpy as np

class DE_Dynamic_ADE_Optimizer:
    def __init__(self, budget, dim, population_size=50, crossover_prob=0.9, cooling_rate=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.cooling_rate = cooling_rate

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        best_solution = population[np.argmin([func(individual) for individual in population])]
        mutation_factors = np.random.uniform(0.5, 1.0, self.population_size)
        crossover_probs = np.full(self.population_size, self.crossover_prob)

        for _ in range(self.budget):
            target_vectors = []
            for i in range(self.population_size):
                indices = np.random.choice([j for j in range(self.population_size) if j != i], 3, replace=False)
                base_vector = population[indices[0]]
                mutated_vector = base_vector + mutation_factors[i] * (population[indices[1]] - population[indices[2]])
                trial_vector = np.where(np.random.rand(self.dim) < crossover_probs[i], mutated_vector, population[i])
                
                current_cost = func(population[i])
                trial_cost = func(trial_vector)
                if trial_cost < current_cost:
                    population[i] = trial_vector
                    if trial_cost < func(best_solution):
                        best_solution = trial_vector
                        mutation_factors[i] *= 1.2
                        crossover_probs[i] *= 1.2  # Increase crossover probability for successful individuals
                else:
                    population[i] = np.clip(0.5 * (trial_vector + population[i]), -5.0, 5.0)
                    mutation_factors[i] *= 0.9
                    crossover_probs[i] *= 0.9  # Decrease crossover probability for unsuccessful individuals

        return best_solution