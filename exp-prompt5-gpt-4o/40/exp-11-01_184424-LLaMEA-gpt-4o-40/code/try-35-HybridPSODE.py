import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.inertia_weight = 0.9  # Adjusted for more exploration initially
        self.cognitive_coefficient = 1.2 + 0.3 * np.random.rand()  # Adaptive cognitive coefficient
        self.social_coefficient = 1.2 + 0.3 * np.random.rand()  # Adaptive social coefficient
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9

    def __call__(self, func):
        lower_bound = -5.0
        upper_bound = 5.0
        pop = np.random.uniform(low=lower_bound, high=upper_bound, size=(self.population_size, self.dim))
        vel = np.random.uniform(low=-1, high=1, size=(self.population_size, self.dim))
        personal_best = np.copy(pop)
        personal_best_values = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_values)]
        
        eval_count = self.population_size
        
        while eval_count < self.budget:
            new_pop = []
            for i in range(self.population_size):
                # Particle Swarm Optimization Update
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                vel[i] = (self.inertia_weight * vel[i] +
                          self.cognitive_coefficient * r1 * (personal_best[i] - pop[i]) +
                          self.social_coefficient * r2 * (global_best - pop[i]))
                candidate = pop[i] + vel[i]
                candidate = np.clip(candidate, lower_bound, upper_bound)
                
                # Differential Evolution Mutation with enhanced dynamic factor
                self.mutation_factor = 0.7 + 0.5 * (1 - eval_count/self.budget)  # Enhanced adaptive mutation factor
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), lower_bound, upper_bound)

                # Crossover with elite preservation
                crossover_mask = np.random.rand(self.dim) < (self.crossover_rate * (1 - eval_count/self.budget))
                candidate = np.where(crossover_mask, mutant, candidate)

                new_pop.append(candidate)

                # Dynamic inertia weight adjustment
                self.inertia_weight = 0.9 - 0.8 * (eval_count/self.budget)**2  # Slightly more aggressive decrement

            # Chaotic initialization and random perturbation to enhance diversity
            if eval_count % 50 == 0:
                perturbation = 0.05 * np.random.uniform(-1, 1, size=(self.population_size, self.dim))  # Reduced perturbation
                new_pop += perturbation

            # Evaluate new population
            new_pop = np.array(new_pop)
            new_values = np.array([func(ind) for ind in new_pop])
            eval_count += self.population_size

            # Update personal and global bests
            improved = new_values < personal_best_values
            personal_best[improved] = new_pop[improved]
            personal_best_values[improved] = new_values[improved]

            best_idx = np.argmin(new_values)
            if new_values[best_idx] < func(global_best):
                global_best = new_pop[best_idx]  # Direct assignment for elite preservation

            pop = new_pop

        return global_best