import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score
import plotly.express as px



class EDA():
    def __init__(self,df):
        self.data = df
    
    def histogram(self,column,bins=10):
        return plt.hist(self.data[column],bins=bins)

    def boxplot(self,column):
        return sns.boxplot(self.data[column])

    def heatmap(self):
        return sns.heatmap(self.data.corr(numeric_only=True),cmap="RdBu")

class Preprocessing():
    def __init__(self,df):
        self.data = df

    def impute(self,column,value):
        self.data.loc[self.data[column].isna(), column] = value

    def categorize(self,column,percentage):
        min = self.data[column].describe().loc["min"]
        max = self.data[column].describe().loc["max"]
        threshold = self.data[column].describe().loc[percentage]
        self.data[column] = pd.DataFrame(pd.cut(self.data[column], [min,threshold,max], labels=[0,1]))
        print(self.data[column])

    def rename(self,current,wanted):
        self.data[wanted] =  self.data[current]
        self.data.drop([current],axis=1)
    
    def log_transform(self,column):
        self.data.loc[:, column] = np.log(self.data[column])
    
    def exp_transform(self,column):
        self.data.loc[:, column] = np.exp(self.data[column])
    
class FeatureEngineering():
    def __init__(self,df):
        self.data = df

    def pca(self,c):
        pca = PCA(n_components=c,random_state=0)
        self.data = pd.DataFrame(pca.fit_transform(self.data))
        return print(pca.explained_variance_)

class ModellingKMeans():

    def __init__(self,df):
        self.data = df

    def check_wcss(self):
        wcss = []
        for i in range(9):
            model = KMeans(random_state=0,n_clusters=i+1,n_init='auto')
            model.fit(self.data)
            wcss.append(model.inertia_)
        return plt.plot(wcss)

    def elbow_visualizer(self):
        model = KMeans(init="k-means++",random_state = 0,n_init='auto')
        visualizer = KElbowVisualizer(model, k = (1,10))
        visualizer.fit(self.data)
        return visualizer.show()
    
    def finalize_model(self,clusters):
        self.model = KMeans(n_clusters=clusters,n_init="auto",random_state=0)
        self.model.fit(self.data)
        score = silhouette_score(self.data, self.model.labels_,random_state=0)
        print("Silhuette score for kmeans model with " +str(clusters)+ " clusters using scaled data: "+str(score))
        return self.model

    def visualize_2d(self):
        pca = PCA(n_components=2,random_state=0)
        df_2d = pd.DataFrame(pca.fit_transform(self.data),columns=["x","y"])
        centers = pca.transform(self.model.cluster_centers_)
        fig, ax = plt.subplots()

        ax.scatter(df_2d["x"],df_2d["y"],c=self.model.labels_,cmap="Set1")

        ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.7);

        return print(fig)

    def visualize_3d(self):
        pca = PCA(n_components=3,random_state=0)
        df_3d = pd.DataFrame(pca.fit_transform(self.data),columns=["x","y","z"])
        fig = px.scatter_3d(df_3d,x="x",y="y",z="z",color=self.model.labels_)
        return fig.show()




