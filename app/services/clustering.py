from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
import pandas as pd
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io

def run_clustering(data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    results = {}
    for i, algorithm_config in enumerate(config['values']):
        algorithm_name = algorithm_config['algorithm']
        parameters = algorithm_config['parameters']
        model_name = f"{algorithm_name} {i + 1}"
        
        if algorithm_name == 'K-Means':
            model = KMeans(**parameters)
        elif algorithm_name == 'DBSCAN':
            model = DBSCAN(**parameters)
        elif algorithm_name == 'Agglomerative Clustering':
            model = AgglomerativeClustering(**parameters)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm_name}")
        
        model.fit(data)
        clusters = model.labels_
        unique_clusters = set(clusters)

        cluster_results = {
            'clusters': {},
            'noise_points': sum([1 for label in clusters if label == -1])
        }

        for cluster in unique_clusters:
            if cluster == -1:
                continue

            cluster_data = data[clusters == cluster]
            cluster_centroid = None

            if algorithm_name == 'K-Means':
                cluster_centroid = model.cluster_centers_[cluster].tolist()
            
            cluster_averages = cluster_data.mean(axis=0).to_dict()
            cluster_size = len(cluster_data)
            cluster_results['clusters'][f"Cluster {cluster}"] = {
                'size': cluster_size,
                'averages': cluster_averages,
                'centroid': cluster_centroid
            }

        # Add PCA results to the cluster results
        cluster_results['pcaData'] = get_pca_data(data, clusters)

        results[model_name] = cluster_results

    return results

def get_pca_data(data: pd.DataFrame, clusters: pd.Series) -> Dict[str, Any]:
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    pca_df = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
    pca_df['Cluster'] = clusters
    print(pca_df)
    return_dict =  {
        'PCA1': pca_df['PCA1'].tolist(),
        'PCA2': pca_df['PCA2'].tolist(),
        'Cluster': pca_df['Cluster'].tolist()
    }
    return return_dict



