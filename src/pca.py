from sklearn.decomposition import PCA
from ggplot import *

from data import data, sound_categories, plot_path

feature_columns = data.columns[1:]

data['label'] = [sound_categories[i] for i in data['category'].tolist()]

# Disregard music category to see if anything changes
data = data[data.label.str.contains("street_music") == False]

pca = PCA(n_components=2)
pca_result = pca.fit_transform(data[feature_columns])

data['pca-one'] = pca_result[:,0]
data['pca-two'] = pca_result[:,1] 

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

chart = ggplot(data, aes(x='pca-one', y='pca-two', color='label')) \
        + geom_point(size=75, alpha=0.8) \
        + ggtitle("First and Second Principal Components coloured by label")

chart.show()
chart.save(plot_path + "principle_component_analysis.png")