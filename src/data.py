import os, json
import pandas as pd
from pprint import pprint

file_names = []
file_paths = []
sound_categories = []

data_path = os.path.dirname(os.path.realpath(__file__)) + '/../extractedFeatures/'

count = 0

# Load .json files using only these numerical keys:
# Each of which has a stat subkey, like 'min', 'max', etc.
# Results to 222 dimensions total in the final transformed dataset

# ['spectral_complexity',
#  'silence_rate_20dB',
#  'average_loudness',
#  'spectral_rms',
#  'spectral_kurtosis',
#  'barkbands_kurtosis',
#  'spectral_spread',
#  'pitch',
#  'dissonance',
#  'spectral_energyband_high',
#  'spectral_skewness',
#  'spectral_flux',
#  'silence_rate_30dB',
#  'spectral_energyband_middle_high',
#  'barkbands_spread',
#  'spectral_centroid',
#  'pitch_salience',
#  'silence_rate_60dB',
#  'spectral_rolloff',
#  'spectral_energyband_low',
#  'barkbands_skewness',
#  'pitch_instantaneous_confidence',
#  'spectral_energyband_middle_low',
#  'spectral_strongpeak',
#  'spectral_decrease',
#  'spectral_energy',
#  'spectral_flatness_db',
#  'zerocrossingrate',
#  'hfc',
#  'spectral_crest'
#  'inharmonicity',
#  'pitch_max_to_total',
#  'pitch_centroid',
#  'pitch_min_to_total',
#  'pitch_after_max_to_before_max_energy_ratio',
#  'oddtoevenharmonicenergyratio']


# Simpler this way. Will deal with other higher dimension keys in dissertation

rows = []

for root, sub_dirs, files in os.walk(data_path):
    for file in files:
        if (file.endswith('.json')):
            if count < 3:
                file_names.append(file)

                path = os.path.join(root, file)
                file_paths.append(path)

                category = root.split('/')[-1]

                if (category not in sound_categories):
                    sound_categories.append(category)

                with open(path) as opened_file:
                    dict_json = json.load(opened_file)

                    # Remove (currently) unusable keys
                    for key in ['tonal', 'rhythm', 'metadata', 'barkbands', 'sccoeffs', 'scvalleys', 'mfcc']:
                        dict_json.pop(key, None)

                    for key in ['barkbands', 'sccoeffs', 'scvalleys', 'mfcc']:
                        dict_json['lowLevel'].pop(key, None)

                    del dict_json['sfx']['tristimulus']

                    # normalized = json_normalize(dict_json)
                    # normalized['category'] = category
                    
                    # print(normalized)
                    rows.append(dict_json)

            count += 1

data = pd.io.json.json_normalize(rows)
