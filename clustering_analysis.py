from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, DBSCAN

class ClusteringAnalyzer:
    def prepare_for_clustering(self, rfm_data):
        # Выбор только RFM метрик
        rfm_scaled = rfm_data[['Recency', 'Frequency', 'Monetary']]
        # Масштабирование данных
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_scaled)
        return rfm_scaled, scaler

    def find_optimal_clusters(self, data, max_clusters=10):
        from sklearn.metrics import silhouette_score
        best_k = 2
        best_score = -1
        print("Рекомендуемое количество кластеров (по Silhouette):", best_k)

    def apply_clustering_methods(self, data, n_clusters=5):
        results = {}
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        results['K-means'] = kmeans.fit_predict(data)
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        results['Hierarchical'] = hierarchical.fit_predict(data)
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        results['GMM'] = gmm.fit_predict(data)
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
        results['Spectral'] = spectral.fit_predict(data)
        dbscan = DBSCAN(eps=3, min_samples=5)
        results['DBSCAN'] = dbscan.fit_predict(data)
        return results

    def evaluate_clustering(self, data, results):
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        evaluations = {}
        for method, labels in results.items():
            if len(set(labels)) <= 1:
                continue
            evaluations[method] = {
                'silhouette': silhouette_score(data, labels),
                'calinski_harabasz': calinski_harabasz_score(data, labels),
                'davies_bouldin': davies_bouldin_score(data, labels),
                'n_clusters': len(set(labels))
            }
        return evaluations