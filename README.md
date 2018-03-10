# Classifying urban sounds with Random Forest

### Run:
1. `python src/pca.py` or `python src/t-sne.py` to explore the data, plots in [plots/](plots/).
2. `python src/random-forest.py` to classify data and get classification results + plots

### (Optional) Extract data from audio with [Essentia](http://essentia.upf.edu/documentation/) (macOS only, already done, files in [extractedFeatures/](extractedFeatures/)):
3. Using [Homebrew](https://brew.sh/): `brew tap MTG/essentia && brew install essentia`
4. `python2.7 extract.py`

## Plots
![t-SNE](https://media.giphy.com/media/XoyZ0yZA9KpvdBNYoO/giphy.gif)

### t-SNE HD video
[![t-SNE higher quality video](https://img.youtube.com/vi/1hJU5p46lAU/0.jpg)](https://www.youtube.com/watch?v=1hJU5p46lAU)

### Principe Component Analysis
![PCA](plots/principle_component_analysis.png)

![](https://media.giphy.com/media/1qk2jBtvdbmazDO68K/giphy.gif)

![](https://media.giphy.com/media/WgN6RNFUdstuljWTU3/giphy.gif)

![Confusion Matrix 0](plots/confusion_matrix_split0.png)

![Feature Importance 0](plots/feature_importances_split0.png)

### Most Important Features
![Most Important Features](plots/most_important_features.png)

