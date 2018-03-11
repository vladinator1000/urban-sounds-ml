from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import operator

from data import data, sound_categories, plot_path

model = RandomForestClassifier()
KFold = StratifiedKFold(n_splits=6, shuffle=True)

# Features without category
features = data.columns[1:]

name_mapping = { key: value for (key, value) in enumerate(sound_categories) }

train_scores = []
test_scores = []
feature_importances = []

# Cross-validation train and test
split_number = 0
for train_indices, test_indices in KFold.split(data, data['category'].tolist()):
    train = data.loc[train_indices]
    test = data.loc[test_indices]

    model.fit(train[features], train['category'].tolist())
    
    train_scores.append(model.score(train[features], train['category'].tolist()))
    test_scores.append(model.score(test[features], test['category'].tolist()))
    
    predicted = model.predict(test[features])
    actual = test['category']

    predicted_categories = [sound_categories[i] for i in predicted]
    actual_categories = [sound_categories[i] for i in actual]

    # Feature Importances http://gph.is/2IdEH9v
    f1, ax1 = plt.subplots(1, 1)
    f1.set_size_inches(40, 40)

    feature_importances.append(model.feature_importances_)
    sns.barplot(model.feature_importances_, list(train[features]), ax=ax1)
    f1.savefig('{}feature_importances_split{}.png'.format(plot_path, split_number))
    
    # Confusion Matrix http://gph.is/2p08Ekt
    crosstab = pd.crosstab(actual, predicted, rownames=['Actual Categories'], colnames=['Predicted Categories'])
    crosstab.replace({ 'Actual Categories': name_mapping, 'Predicted Categories': name_mapping })

    f2, ax2 = plt.subplots(1, 1)
    f2.set_size_inches(15, 15)

    sns.heatmap(crosstab, ax=ax2)
    ax2.set_xticklabels(sound_categories)
    ax2.set_yticklabels(sound_categories)
    ax2.set_title('Confusion Matrix')

    for tick in ax2.get_yticklabels():
        tick.set_rotation(0)

    f2.savefig('{}confusion_matrix_split{}.png'.format(plot_path, split_number))

    split_number += 1

print('\nSummary after k-fold validation:')
print('Training Scores: {}'.format(stats.describe(train_scores)))
print('Testing Scores: {}'.format(stats.describe(test_scores)))

feature_importances = pd.DataFrame(feature_importances, columns=list(data[features]))
feature_importance_means_dict = feature_importances.mean().to_dict()
sorted_importance_means = sorted(feature_importance_means_dict.items(), key=operator.itemgetter(1)) 

most_important = sorted_importance_means[-10:]
print('\nMost important features: {}'.format(most_important))

f3, ax3 = plt.subplots(1, 1)
f3.set_size_inches(20, 20)
ax3.set_title('Most Important Features')

for tick in ax3.get_xticklabels():
    tick.set_rotation(45)

sns.barplot(*zip(*most_important), ax=ax3)

f3.savefig('{}most_important_features.png'.format(plot_path))

# Summary after k-fold validation:
# Training Scores: DescribeResult(nobs=6, minmax=(0.9888143176733781, 1.0), mean=0.9948179386894326, variance=1.3361562926299023e-05, skewness=-0.3546967651392324, kurtosis=-0.27651987199855155)
# Testing Scores: DescribeResult(nobs=6, minmax=(0.5625, 0.7191011235955056), mean=0.6324522226363066, variance=0.0037360774721599288, skewness=0.2206773149797684, kurtosis=-1.3230495066810048)

# Most important features: [
#     'spectral_kurtosis.max',
#     'spectral_skewness.max',
#     'pitch_salience.var',
#     'spectral_kurtosis.var',
#     'pitch_instantaneous_confidence.max',
#     'pitch_salience.mean',
#     'spectral_energyband_low.median',
#     'spectral_flatness_db.var',
#     'spectral_skewness.var',
#     'pitch_instantaneous_confidence.var'
# ]