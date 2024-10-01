import numpy as np
import scanpy as sc
from scib import metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def calculate_map(adata, compound_key='Metadata_InChIKey', negative_control='IAZDPXIOMUYVGZ-UHFFFAOYSA-N', use_negative_controls=True):
    print('Calculating mAP...')
    compounds = adata.obs[compound_key].unique()
    ap_scores = []

    print(f"Total number of compounds: {len(compounds)}")
    print(f"Compound key: {compound_key}")
    print(f"Negative control: {negative_control}")
    print(f"Use negative controls: {use_negative_controls}")

    for compound in compounds:
        if compound == negative_control:
            continue

        print(f"Processing compound: {compound}")

        # Get query samples (all samples of the current compound)
        query_samples = adata[adata.obs[compound_key] == compound]
        print(f"Number of query samples: {len(query_samples)}")
        
        # Get all samples (including the current compound)
        all_samples = adata.copy()
        
        if not use_negative_controls:
            # Exclude negative controls if not using them
            all_samples = all_samples[all_samples.obs[compound_key] != negative_control]
        
        print(f"Number of all samples: {len(all_samples)}")
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_samples.X, all_samples.X)
        print(f"Shape of similarities matrix: {similarities.shape}")
        
        # Calculate AP for each query
        for i, query_similarity in enumerate(similarities):
            sorted_indices = np.argsort(query_similarity)[::-1]
            relevant_ranks = np.where(all_samples.obs[compound_key].iloc[sorted_indices] == compound)[0] + 1
            
            print(f"Query {i+1}: Number of relevant ranks: {len(relevant_ranks)}")
            
            if len(relevant_ranks) == 0:
                print(f"Warning: No relevant samples found for query {i+1} of compound {compound}")
                continue
            
            precisions = np.arange(1, len(relevant_ranks) + 1) / relevant_ranks
            ap = np.mean(precisions)
            ap_scores.append(ap)
            print(f"AP score for query {i+1}: {ap}")
    
    final_map = np.mean(ap_scores) if ap_scores else 0
    print(f"Final mAP score: {final_map}")
    print(f"Number of AP scores: {len(ap_scores)}")
    return final_map

def calculate_metrics(adata, label_key, batch_key, compound_key='Metadata_InChIKey', negative_control='IAZDPXIOMUYVGZ-UHFFFAOYSA-N'):
    """Calculate metrics shown in the image table for a given AnnData object."""
    sc.pp.neighbors(adata, use_rep='X', n_neighbors=25, metric='cosine')
    sc.tl.leiden(adata, key_added='Metadata_Cluster')
    
    # Ensure label_key and batch_key columns are of category dtype
    if not adata.obs[label_key].dtype.name == 'category':
        adata.obs[label_key] = adata.obs[label_key].astype('category')
    if not adata.obs[batch_key].dtype.name == 'category':
        adata.obs[batch_key] = adata.obs[batch_key].astype('category')
    
    # Store the main data matrix as an embedding in obsm
    adata.obsm['X_pca'] = adata.X
    
    # Calculate scib metrics
    scib_results = metrics(
        adata,
        adata,
        batch_key=batch_key,
        label_key=label_key,
        cluster_key='Metadata_Cluster',
        embed='X_pca',  # Use the stored embedding
        isolated_labels_asw_=True,
        silhouette_=True,
        hvg_score_=True,
        graph_conn_=True,
        pcr_=True,
        kBET_=False,
        nmi_=True,
        ari_=True,
        ilisi_=False,
        clisi_=False,
    )

    print(scib_results)
    
    # Map scib metric names to our metric names
    metric_mapping = {
        'graph_conn': 'graph_connectivity',
        'ASW_label/batch': 'silhouette_batch',
        'NMI_cluster/label': 'leiden_NMI',
        'ARI_cluster/label': 'leiden_ARI',
        'ASW_label': 'silhouette_label',
    }
    
    results = {}
    for scib_name, our_name in metric_mapping.items():
        if scib_name in scib_results.index:
            results[our_name] = scib_results.loc[scib_name, 0]
        else:
            print(f"Warning: Metric '{scib_name}' not found in scib_results. Setting to 1e-9.")
            results[our_name] = 1e-9
    
    # Calculate additional metrics not provided by scib
    results['mAP_controls'] = calculate_map(adata, compound_key, negative_control, use_negative_controls=True)
    results['mAP_nonmix'] = calculate_map(adata, compound_key, negative_control, use_negative_controls=False)
    
    # Calculate custom metrics
    custom_metrics = calculate_custom_metrics(adata, compound_key, negative_control)
    results.update(custom_metrics)
    
    # Calculate aggregate scores
    batch_correction_metrics = ['graph_connectivity', 'silhouette_batch', 'classification_batch_without_nc']
    bio_metrics = ['leiden_NMI', 'leiden_ARI', 'silhouette_label', 'mAP_controls', 'mAP_nonmix', 'classification_label_without_nc']
    
    results['aggregate_batch_correction'] = np.mean([results[metric] for metric in batch_correction_metrics])
    results['aggregate_bio_metrics'] = np.mean([results[metric] for metric in bio_metrics])
    results['aggregate_overall'] = np.mean([results['aggregate_batch_correction'], results['aggregate_bio_metrics']])
    
    return results

def calculate_custom_metrics(adata, compound_key='Metadata_InChIKey', negative_control='IAZDPXIOMUYVGZ-UHFFFAOYSA-N'):
    data = adata.X
    metadata = adata.obs

    # Classification metrics with negative control
    classification_label_with_nc = perform_classification(data, metadata[compound_key])
    classification_batch_with_nc = 1 - perform_classification(data, metadata['Metadata_Batch'])

    # Classification metrics without negative control
    adata_without_nc = adata[adata.obs[compound_key] != negative_control].copy()
    data_without_nc = adata_without_nc.X
    metadata_without_nc = adata_without_nc.obs
    classification_label_without_nc = perform_classification(data_without_nc, metadata_without_nc[compound_key])
    classification_batch_without_nc = 1 - perform_classification(data_without_nc, metadata_without_nc['Metadata_Batch'])

    # Noise robustness
    noise_robustness_with_nc = test_robustness_to_noise(data, metadata[compound_key])
    noise_robustness_without_nc = test_robustness_to_noise(data_without_nc, metadata_without_nc[compound_key])

    return {
        'classification_label_with_nc': classification_label_with_nc,
        'classification_batch_with_nc': classification_batch_with_nc,
        'classification_label_without_nc': classification_label_without_nc,
        'classification_batch_without_nc': classification_batch_without_nc,
        'noise_robustness_with_nc': noise_robustness_with_nc,
        'noise_robustness_without_nc': noise_robustness_without_nc
    }

def perform_classification(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return f1

def test_robustness_to_noise(data, labels, noise_levels=[0.1, 0.2, 0.3]):
    base_f1 = perform_classification(data, labels)
    f1_differences = []
    print(f"Base F1 score: {base_f1}")
    
    for noise_level in noise_levels:
        noisy_data = data + np.random.normal(0, noise_level, data.shape)
        noisy_f1 = perform_classification(noisy_data, labels)
        f1_differences.append(base_f1 - noisy_f1)
        print(f"F1 score with noise level {noise_level}: {noisy_f1}")
    
    avg_f1_difference = np.mean(f1_differences)
    robustness_score = 1 - avg_f1_difference
    
    print(f"Robustness score: {robustness_score}")
    return robustness_score