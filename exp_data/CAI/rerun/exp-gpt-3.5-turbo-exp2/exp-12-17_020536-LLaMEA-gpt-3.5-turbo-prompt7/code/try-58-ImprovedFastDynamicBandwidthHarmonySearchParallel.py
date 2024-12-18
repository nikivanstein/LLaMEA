class ImprovedFastDynamicBandwidthHarmonySearchParallel(FastDynamicBandwidthHarmonySearch):
    def __call__(self, func):
        pitch_rate = 0.45
        bandwidth = 0.1
        parallel_points = 5
        for _ in range(self.budget // parallel_points):
            new_harmonies = np.random.uniform(self.lower_bound, self.upper_bound, (parallel_points, self.dim))
            for _ in range(2):
                pitch_rate = max(0.01, pitch_rate * 0.97 * 1.3 * (1 + 0.1 * np.mean(np.abs(self.harmony_memory[0] - self.harmony_memory[-1])))
                for i in range(self.dim):
                    if np.random.rand() < pitch_rate:
                        new_harmonies[:, i] = np.random.choice(self.harmony_memory[:, i], parallel_points)
            func_values = [func(h) for h in new_harmonies]
            best_index = np.argmin(func_values)
            if func_values[best_index] < func(self.harmony_memory[-1]):
                self.harmony_memory[-1] = new_harmonies[best_index]
                self.harmony_memory = self.harmony_memory[np.argsort([func(h) for h in self.harmony_memory])]
            for _ in range(2):
                for i in range(self.dim):
                    new_harmonies_local = np.copy(self.harmony_memory[-1])
                    new_harmonies_local[:, i] = np.random.uniform(max(self.lower_bound, self.harmony_memory[-1][i] - bandwidth),
                                                             min(self.upper_bound, self.harmony_memory[-1][i] + bandwidth), parallel_points)
                    func_values_local = [func(h) for h in new_harmonies_local]
                    best_local_index = np.argmin(func_values_local)
                    if func_values_local[best_local_index] < func(self.harmony_memory[-1]):
                        self.harmony_memory[-1] = new_harmonies_local[best_local_index]
            for i in range(self.dim):
                new_harmonies_pitch = np.copy(self.harmony_memory[-1])
                new_harmonies_pitch[:, i] = np.random.uniform(max(self.lower_bound, self.harmony_memory[-1][i] - bandwidth),
                                                         min(self.upper_bound, self.harmony_memory[-1][i] + bandwidth), parallel_points)
                func_values_pitch = [func(h) for h in new_harmonies_pitch]
                best_pitch_index = np.argmin(func_values_pitch)
                if func_values_pitch[best_pitch_index] < func(self.harmony_memory[-1]):
                    self.harmony_memory[-1] = new_harmonies_pitch[best_pitch_index]
            bandwidth = min(0.5, max(0.01, bandwidth * (1 + 0.1 * np.mean(np.abs(self.harmony_memory[0] - self.harmony_memory[-1])))))
        return self.harmony_memory[0]