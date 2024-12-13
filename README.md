[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/ewMf7vy3)
import numpy as np
import pandas as pd

# داده‌ها (شهرها و مختصات)
data = {
    'City': ['Tehran', 'Isfahan', 'Tabriz', 'Shiraz', 'Mashhad',
             'Kermanshah', 'Yazd', 'Karaj', 'Ahvaz', 'Qom',
             'Urmia', 'Arak', 'Kerman', 'Zanjan', 'Sari',
             'Gorgan', 'Bandar Abbas', 'Birjand', 'Sabzevar', 'Bojnurd'],
    'Latitude': [35.6892, 32.6546, 38.0962, 29.5918, 36.2605,
                 34.3293, 31.8974, 35.8325, 31.3193, 34.639,
                 37.5505, 34.0944, 30.2835, 36.6732, 36.6751,
                 36.6975, 27.1884, 32.8716, 33.0565, 37.4875],
    'Longitude': [51.3890, 51.6570, 46.2913, 52.5836, 59.5443,
                  47.1167, 54.3660, 51.9792, 48.6692, 50.8764,
                  45.9773, 49.6957, 57.0786, 48.5014, 53.0204,
                  54.0172, 56.2167, 59.2253, 57.6775, 57.1847]
}

iran_df = pd.DataFrame(data)
points = list(zip(iran_df['Longitude'], iran_df['Latitude']))

# الگوریتم ژنتیک
class GeneticAlgorithm:
    def __init__(self, points, population_size=100, generations=500, mutation_rate=0.01):
        self.points = points
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()

    def initialize_population(self):
        return [np.random.permutation(len(self.points)) for _ in range(self.population_size)]

    def calculate_distance(self, tour):
        distance = 0
        for i in range(len(tour)):
            point1 = self.points[tour[i]]
            point2 = self.points[tour[(i + 1) % len(tour)]]
            distance += np.linalg.norm(np.array(point1) - np.array(point2))
        return distance

    def select_parents(self):
        # روش انتخاب مسابقه‌ای
        tournament_size = 5
        tournament = np.random.choice(self.population, size=tournament_size)
        distances = [self.calculate_distance(tour) for tour in tournament]
        best_parent1 = tournament[np.argmin(distances)]
        
        tournament = np.random.choice(self.population, size=tournament_size)
        distances = [self.calculate_distance(tour) for tour in tournament]
        best_parent2 = tournament[np.argmin(distances)]
        
        return best_parent1, best_parent2

    def crossover(self, parent1, parent2):
        start, end = sorted(np.random.choice(len(parent1), 2, replace=False))
        child = [None] * len(parent1)
        child[start:end + 1] = parent1[start:end + 1]
        current_position = (end + 1) % len(parent1)
        for gene in parent2:
            if gene not in child:
                child[current_position] = gene
                current_position = (current_position + 1) % len(parent1)
        return np.array(child)

    def mutate(self, tour):
        if np.random.rand() < self.mutation_rate:
            idx1, idx2 = np.random.choice(len(tour), 2, replace=False)
            tour[idx1], tour[idx2] = tour[idx2], tour[idx1]

    def run(self):
        for generation in range(self.generations):
            new_population = []
            for _ in range(self.population_size):
                parent1, parent2 = self.select_parents()
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)

            self.population = new_population

        best_tour = min(self.population, key=self.calculate_distance)
        best_distance = self.calculate_distance(best_tour)
        return best_tour, best_distance

ga = GeneticAlgorithm(points)
best_tour, best_distance = ga.run()

# نمایش بهترین مسیر
cities = iran_df['City'].tolist()
print("Best tour:", [cities[i] for i in best_tour])
print("Best distance:", best_distance)
