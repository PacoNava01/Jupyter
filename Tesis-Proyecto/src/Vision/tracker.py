import numpy as np
from scipy.spatial import distance

class CentroidTracker:
    def __init__(self, max_disappeared=10):
        self.next_id = 0
        self.objects = {}        # id → (x, y)
        self.disappeared = {}    # id → frames sin verse
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, input_centroids):
        # si no hay detecciones
        if len(input_centroids) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        # si no hay objetos previos
        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c)
            return self.objects

        # IDs y posiciones actuales
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        # matriz de distancias
        D = distance.cdist(np.array(object_centroids), input_centroids)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.disappeared[object_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        # objetos no asignados
        unused_rows = set(range(D.shape[0])) - used_rows
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1

            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        # nuevos objetos
        unused_cols = set(range(D.shape[1])) - used_cols
        for col in unused_cols:
            self.register(input_centroids[col])

        return self.objects