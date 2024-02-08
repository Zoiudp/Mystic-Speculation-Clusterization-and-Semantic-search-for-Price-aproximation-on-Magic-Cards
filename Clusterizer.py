from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import pandas as pd

noatributes = ['originalType','text','price','rarity','pred_price']

def MCluster(dfs):
    features = dfs.drop(columns= [col for col in dfs if col in noatributes])
    target = dfs['price']

    valores = [[3, 2, 4, 2, 0]]

    test_input = pd.DataFrame(valores, columns=['manaValue', 'power', 'toughness', 'raridade', 'classified_text'])

    label_encoder = LabelEncoder()

    true_labels = label_encoder.fit_transform(target)

    n_clusters = len(label_encoder.classes_)

    preprocessor = Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("pca", PCA(n_components=5, random_state=42)),
        ]
    )

    clusterer = Pipeline(
    [
        (
            "kmeans",
            KMeans(
                n_clusters=n_clusters,
                init="k-means++",
                n_init=50,
                max_iter=500,
                random_state=42,
            ),
         ),
    ])

    pipe = Pipeline(
     [
         ("preprocessor", preprocessor),
         ("clusterer", clusterer)
     ])
    
    pipe.fit(features)

    preprocessed_data = pipe["preprocessor"].transform(features)

    predicted_labels = pipe["clusterer"]["kmeans"].labels_

    print(silhouette_score(preprocessed_data, predicted_labels))

    dfcluster = pd.DataFrame(pipe["preprocessor"].transform(features),columns=features.columns,)

    dfcluster["predicted_cluster"] = pipe["clusterer"]["kmeans"].labels_
    dfcluster["true_label"] = label_encoder.inverse_transform(true_labels)

    print(pipe.predict(test_input))

    dfcluster['true_label'].loc[dfcluster['predicted_cluster'] == int(pipe.predict(test_input))]

    #valor final previsto pra carta.
    print(dfcluster['true_label'].loc[dfcluster['predicted_cluster'] == int(pipe.predict(test_input))].mean())


