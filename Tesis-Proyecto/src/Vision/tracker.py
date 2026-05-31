import numpy as np
from scipy.spatial import distance

'''
Pendiente por dictaminar su eliminación
'''

class CentroidTracker:
    def __init__(self, max_disappeared=10, max_distance=100):
        """
        max_disappeared: frames antes de eliminar objeto
        max_distance: distancia máxima para asociar detecciones
        """
        self.next_object_id = 0
        self.objects = {}          # object_id -> centroid (x, y)
        self.disappeared = {}      # object_id -> frames sin verse

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    # -----------------------------------------------------
    # REGISTRAR NUEVO OBJETO
    # -----------------------------------------------------
    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    # -----------------------------------------------------
    # ELIMINAR OBJETO
    # -----------------------------------------------------
    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    # -----------------------------------------------------
    # ACTUALIZAR TRACKING
    # -----------------------------------------------------
    def update(self, input_centroids):
        """
        input_centroids: lista de (x, y)
        """

        # ---------------------------------
        # CASO 1: no hay detecciones
        # ---------------------------------
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            return self.objects

        input_centroids = np.array(input_centroids)

        # ---------------------------------
        # CASO 2: no hay objetos activos
        # ---------------------------------
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(tuple(centroid))
            return self.objects

        # ---------------------------------
        # CASO 3: matching
        # ---------------------------------
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        object_centroids_np = np.array(object_centroids)

        # matriz de distancias
        D = distance.cdist(object_centroids_np, input_centroids)

        # ordenar por menor distancia
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        # asociación
        for row, col in zip(rows, cols):

            if row in used_rows or col in used_cols:
                continue

            if D[row, col] > self.max_distance:
                continue

            object_id = object_ids[row]
            self.objects[object_id] = tuple(input_centroids[col])
            self.disappeared[object_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        # ---------------------------------
        # objetos no usados (desaparecen)
        # ---------------------------------
        unused_rows = set(range(len(object_centroids))) - used_rows

        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1

            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        # ---------------------------------
        # nuevos objetos
        # ---------------------------------
        unused_cols = set(range(len(input_centroids))) - used_cols

        for col in unused_cols:
            self.register(tuple(input_centroids[col]))

        return self.objects