from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from data import data, sound_categories, plot_path

model = RandomForestClassifier()
KFold = StratifiedKFold(n_splits = 6, shuffle = True)

# Features without category
features = data.columns[1:]

name_mapping = { key: value for (key, value) in enumerate(sound_categories) }
# plt.rcParams["figure.figsize"] = (20, 20)

# Cross-validation train and test
split_number = 0
for train_indices, test_indices in KFold.split(data, data['category'].tolist()):
    train = data.loc[train_indices]
    test = data.loc[test_indices]

    model.fit(train[features], train['category'].tolist())
    predicted = model.predict(test[features])
    actual = test['category']

    predicted_categories = [sound_categories[i] for i in predicted]
    actual_categories = [sound_categories[i] for i in actual]

    # Feature Importances http://gph.is/2IdEH9v
    crosstab = pd.crosstab(actual, predicted, rownames=['Actual Categories'], colnames=['Predicted Categories'])
    crosstab.replace({ 'Actual Categories': name_mapping, 'Predicted Categories': name_mapping })
    
    f1, ax1 = plt.subplots(1, 1)
    f1.set_size_inches(40, 40)

    sns.barplot(model.feature_importances_, list(train[features]), ax=ax1)
    f1.savefig('{}feature_importances_split{}.png'.format(plot_path, split_number))
    
    # Confusion Matrix http://gph.is/2p08Ekt
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
    # plt.show()
