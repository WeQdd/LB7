import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class VisualizationUtils:
    def plot_rfm_distributions(self, rfm_data):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i, col in enumerate(['Recency', 'Frequency', 'Monetary']):
            sns.histplot(rfm_data[col], ax=axes[i], kde=True)
        plt.tight_layout()
        plt.show()

    def plot_clusters_3d(self, rfm_data):
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(rfm_data['Recency'], rfm_data['Frequency'], rfm_data['Monetary'],
                             c=rfm_data['Cluster'], cmap='viridis', s=50)
        plt.title('RFM Кластеры (3D)')
        plt.colorbar(scatter, label='Кластер')
        plt.show()

    def plot_cluster_characteristics(self, rfm_data):
        cluster_means = rfm_data.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
        cluster_means.plot(kind='bar', figsize=(12, 6))
        plt.title('Средние значения RFM по кластерам')
        plt.ylabel('Значение')
        plt.xlabel('Кластер')
        plt.xticks(rotation=0)
        plt.legend(title='Метрика')
        plt.grid(True)
        plt.show()

    def plot_cluster_heatmap(self, rfm_data):
        cluster_means = rfm_data.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
        cluster_means = cluster_means.apply(lambda x: x / x.max(), axis=0)
        plt.figure(figsize=(10, 6))
        sns.heatmap(cluster_means, annot=True, cmap='coolwarm', center=0)
        plt.title('RFM профили кластеров (нормализованные)')
        plt.show()