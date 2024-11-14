import numpy as np

class ImprovedMultiPopDEHarmonySearch(MultiPopDEHarmonySearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.differential_weight = np.random.uniform(0, 2, size=(self.budget, self.dim))
        self.mutation_scale = np.random.uniform(0.1, 0.5, size=(self.budget, self.dim))  # Introducing dynamic mutation scale
    
    def __call__(self, func):
        populations = [np.random.uniform(self.lower_bound, self.upper_bound, size=(self.population_size, self.dim)) for _ in range(self.num_populations)]
        for _ in range(self.budget):
            for pop_idx, population in enumerate(populations):
                for i in range(self.population_size):
                    donor_idx, trial1_idx, trial2_idx = np.random.choice(self.population_size, 3, replace=False)
                    donor = population[donor_idx]
                    trial1 = population[trial1_idx]
                    trial2 = population[trial2_idx]
                    mutant_vector = donor + self.differential_weight[i] * (trial1 - trial2) * self.mutation_scale[i]  # Dynamic mutation strategy based on fitness
                    mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                    if func(mutant_vector) < func(population[i]):
                        population[i] = mutant_vector

            self.step_size *= 0.995  # Dynamic step size adaptation based on individual harmony improvements

        all_populations = np.concatenate(populations)
        return all_populations[np.argmin([func(p) for p in all_populations])]