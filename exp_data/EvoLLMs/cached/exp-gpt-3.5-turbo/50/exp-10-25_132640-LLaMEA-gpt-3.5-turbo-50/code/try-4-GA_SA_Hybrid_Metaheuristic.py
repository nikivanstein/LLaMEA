import numpy as np

class GA_SA_Hybrid_Metaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.max_iter = budget // self.pop_size
        self.mutation_rate = 0.2
        self.sa_steps = 10
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        def clip(x):
            return np.clip(x, self.lb, self.ub)

        def evaluate(x):
            return func(clip(x))

        def create_population():
            return np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        
        def crossover(parent1, parent2):
            crossover_point = np.random.randint(1, self.dim)
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            return child

        def mutate(individual):
            mutated_genes = np.random.uniform(self.lb, self.ub, self.dim)
            mask = np.random.choice([0, 1], size=self.dim, p=[1 - self.mutation_rate, self.mutation_rate])
            individual = individual * (1 - mask) + mutated_genes * mask
            return individual

        def SA_optimization(x):
            current_state = x.copy()
            current_energy = evaluate(current_state)
            best_state = current_state.copy()
            best_energy = current_energy

            for _ in range(self.sa_steps):
                candidate_state = mutate(current_state)
                candidate_energy = evaluate(candidate_state)

                if candidate_energy < current_energy:
                    current_state = candidate_state
                    current_energy = candidate_energy

                    if candidate_energy < best_energy:
                        best_state = candidate_state
                        best_energy = candidate_energy
                else:
                    acceptance_probability = np.exp(-(candidate_energy - current_energy))
                    if np.random.uniform() < acceptance_probability:
                        current_state = candidate_state
                        current_energy = candidate_energy

            return best_state

        population = create_population()
        best_solution = population[0]

        for _ in range(self.max_iter):
            offspring = []

            for _ in range(self.pop_size):
                parent1, parent2 = np.random.choice(population, 2, replace=False)
                child = crossover(parent1, parent2)
                mutated_child = mutate(child)
                offspring.append(mutated_child)

            population = np.array(offspring)
            population = np.array([SA_optimization(individual) for individual in population])

            best_idx = np.argmin([evaluate(individual) for individual in population])
            best_solution = population[best_idx]

        return best_solution