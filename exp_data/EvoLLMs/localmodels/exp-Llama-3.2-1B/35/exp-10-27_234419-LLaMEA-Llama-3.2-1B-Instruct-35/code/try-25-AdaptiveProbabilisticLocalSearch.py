import numpy as np
import random

class AdaptiveProbabilisticLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func = None
        self.search_space = None
        self.sample_size = None
        self.sample_indices = None
        self.local_search = False
        self.crossover_probability = 0.35
        self.mutation_probability = 0.35

    def __call__(self, func):
        if self.func is None:
            self.func = func
            self.search_space = np.random.uniform(-5.0, 5.0, self.dim)
            self.sample_size = 1
            self.sample_indices = None

        if self.budget <= 0:
            raise ValueError("Budget is less than or equal to zero")

        for _ in range(self.budget):
            if self.sample_indices is None:
                self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
            else:
                self.sample_indices = np.random.choice(self.sample_indices, size=self.sample_size, replace=False)
            self.local_search = False

            if self.local_search:
                best_func = func(self.sample_indices)
                if np.abs(best_func - func(self.sample_indices)) < np.abs(func(self.sample_indices) - func(self.sample_indices)):
                    self.sample_indices = None
                    self.local_search = False
                    self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
                    self.sample_indices = self.sample_indices[:self.sample_size]
                else:
                    self.sample_indices = None
                    self.local_search = False

            if self.sample_indices is None:
                best_func = func(self.sample_indices)
                self.sample_indices = None
                self.local_search = False

            if np.abs(best_func - func(self.sample_indices)) < 1e-6:
                break

            # Refine the strategy
            if random.random() < self.crossover_probability:
                parent1, parent2 = self.sample_indices[:self.sample_size//2], self.sample_indices[self.sample_size//2:]
                child = self.mutate(parent1, parent2)
                self.sample_indices = [child]
            else:
                self.sample_indices = None
                self.local_search = False

            if self.sample_indices is None:
                best_func = func(self.sample_indices)
                self.sample_indices = None
                self.local_search = False

            if np.abs(best_func - func(self.sample_indices)) < 1e-6:
                break

        return func(self.sample_indices)

    def mutate(self, parent1, parent2):
        # Mutate the parent1 and parent2 to produce a new individual
        # This can be done using evolutionary crossover or mutation operators
        # For simplicity, we will use a random mutation operator
        if random.random() < self.mutation_probability:
            # Randomly select a gene from parent1 and parent2
            gene1, gene2 = np.random.choice(self.dim, size=1, replace=False)
            # Randomly flip the gene
            if random.random() < 0.5:
                gene1 = 1 - gene1
            # Replace the gene in parent1 and parent2
            parent1[gene1] = 1 - parent1[gene1]
            parent2[gene2] = 1 - parent2[gene2]
        return np.array([gene1, gene2])