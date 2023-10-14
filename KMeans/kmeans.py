import numpy as np

class KMeans:
    def __init__(self, n_clusters=5, initCent='random', max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.initCent = initCent
        self.centroids = None
        self.cluster_assessment = None
        self.labels = None
        self.sse = None

    def _dist_euclidean(self, vecA, vecB):
        return np.linalg.norm(vecA - vecB)

    def _rand_centroids(self, X):
        n = X.shape[1]
        centroids = np.empty((self.n_clusters, n))
        for j in range(n):
            min_j = min(X[:, j])
            range_j = float(max(X[:, j]) - min_j)
            centroids[:, j] = (min_j + range_j * np.random.rand(self.n_clusters))
        return centroids

    def fit(self, X):
        try:
            if not isinstance(X, np.ndarray):
                X = np.asarray(X)
        except:
            raise TypeError("numpy.ndarray required for X")

        m = X.shape[0
        self.cluster_assessment = np.empty((m, 2))
        if self.initCent == 'random':
            self.centroids = self._rand_centroids(X)

        for _ in range(self.max_iter):
            cluster_changed = False
            for i in range(m):
                min_dist = np.inf
                min_index = -1
                for j in range(self.n_clusters):
                    dist_ji = self._dist_euclidean(self.centroids[j, :], X[i, :])
                    if dist_ji < min_dist:
                        min_dist = dist_ji
                        min_index = j
                if self.cluster_assessment[i, 0] != min_index:
                    cluster_changed = True
                    self.cluster_assessment[i, :] = min_index, min_dist ** 2

            if not cluster_changed:
                break
            for i in range(self.n_clusters):
                pts_in_cluster = X[np.where(self.cluster_assessment[:, 0] == i)[0]]
                self.centroids[i, :] = np.mean(pts_in_cluster, axis=0)

        self.labels = self.cluster_assessment[:, 0]
        self.sse = sum(self.cluster_assessment[:, 1])

    def predict(self, X):
        try:
            if not isinstance(X, np.ndarray):
                X = np.asarray(X)
        except:
            raise TypeError("numpy.ndarray required for X")

        m = X.shape[0
        preds = np.empty((m,))
        for i in range(m):
            min_dist = np.inf
            for j in range(self.n_clusters):
                dist_ji = self._dist_euclidean(self.centroids[j, :], X[i, :])
                if dist_ji < min_dist:
                    min_dist = dist_ji
                    preds[i] = j
        return preds

class BiKMeans:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.centroids = None
        self.cluster_assessment = None
        self.labels = None
        self.sse = None

    def _dist_euclidean(self, vecA, vecB):
        return np.linalg.norm(vecA - vecB)

    def fit(self, X):
        try:
            if not isinstance(X, np.ndarray):
                X = np.asarray(X)
        except:
            raise TypeError("numpy.ndarray required for X")

        m = X.shape[0
        self.cluster_assessment = np.zeros((m, 2))
        centroid0 = np.mean(X, axis=0).tolist()
        cent_list = [centroid0]

        for j in range(m):
            self.cluster_assessment[j, 1] = self._dist_euclidean(np.asarray(centroid0), X[j, :]) ** 2

        while len(cent_list) < self.n_clusters:
            lowest_sse = np.inf
            for i in range(len(cent_list)):
                pts_in_curr_cluster = X[np.where(self.cluster_assessment[:, 0] == i)[0], :]
                clf = KMeans(n_clusters=2)
                clf.fit(pts_in_curr_cluster)
                centroid_mat, split_cluster_assessment = clf.centroids, clf.cluster_assessment
                sse_split = sum(split_cluster_assessment[:, 1])
                sse_not_split = sum(self.cluster_assessment[np.where(self.cluster_assessment[:, 0] != i)[0], 1])
                if (sse_split + sse_not_split) < lowest_sse:
                    best_cent_to_split = i
                    best_new_cents = centroid_mat
                    best_cluster_assessment = split_cluster_assessment.copy()
                    lowest_sse = sse_split + sse_not_split
            best_cluster_assessment[np.where(best_cluster_assessment[:, 0] == 1)[0], 0] = len(cent_list)
            best_cluster_assessment[np.where(best_cluster_assessment[:, 0] == 0)[0], 0] = best_cent_to_split
            cent_list[best_cent_to_split] = best_new_cents[0, :].tolist()
            cent_list.append(best_new_cents[1, :].tolist()
            self.cluster_assessment[np.where(self.cluster_assessment[:, 0] == best_cent_to_split)[0], :] = best_cluster_assessment

        self.labels = self.cluster_assessment[:, 0]
        self.sse = sum(self.cluster_assessment[:, 1])
        self.centroids = np.asarray(cent_list)

    def predict(self, X):
        try:
            if not isinstance(X, np.ndarray):
                X = np.asarray(X)
        except:
            raise TypeError("numpy.ndarray required for X")

        m = X.shape[0
        preds = np.empty((m,))
        for i in range(m):
            min_dist = np.inf
            for j in range(self.n_clusters):
                dist_ji = self._dist_euclidean(self.centroids[j, :], X[i, :])
                if dist_ji < min_dist:
                    min_dist = dist_ji
                    preds[i] = j
        return preds
