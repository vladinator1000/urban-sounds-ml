import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE

from data import data, sound_categories, plot_path

feature_columns = data.columns[1:]

tsne = TSNE(n_components=3, random_state=0)
tsne_result = np.array(tsne.fit_transform(data[feature_columns], sound_categories))

palette = np.array(sns.color_palette("hls", len(sound_categories)))
fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(list(tsne_result[:, 0]), list(tsne_result[:, 1]), list(tsne_result[:, 2]), c=palette[data['category']])

markers = [plt.Line2D([0,0],[0,0], color=color, marker='o', linestyle='') for color in palette]
ax.legend(markers, sound_categories, numpoints=1)

print('LMB to pan around, RMB + drag mouse to zoom')

plt.show()