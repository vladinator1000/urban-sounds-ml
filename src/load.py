# Extracts descriptive parameters from urban sounds
# Dataset can be downloaded here (16GB+)
# https://serv.cusp.nyu.edu/projects/urbansounddataset/download-urbansound.html

import essentia, os, errno
from essentia.standard import *

file_names = []
file_paths = []
categories = []
file_extension = '.wav'


# Get the audio directory path
#audio_path = os.path.dirname(os.path.realpath(__file__)) + '/../data/'
audio_path = os.path.dirname("/Volumes/Seagate Backup Plus Drive/uniiii/UrbanSound/data/")
max_files = -1
max_files -= 1

# # Get the names and paths of all .file_extension files in the audio folder
counter = 0
for root, sub_dirs, files in os.walk(audio_path):
    for file in files:
        if(file.endswith(file_extension)):
            # if(max_files < 0 and counter > max_files):
            #     break

            file_names.append(file)
            file_paths.append(os.path.join(root, file))

            category = root.split('/')[-1]

            if category != 'data' and category not in categories:
                categories.append(category)

            counter += 1

# Load them with equal loudness
loaded_files = []
for path in file_paths:
    # Using audio loader of choice http://essentia.upf.edu/documentation/algorithms_overview.html#audio-input-output
    loader = EqloudLoader(filename = path)
    loaded_files.append(loader())

dataPools = []
data_pools_aggregated = []
extractor = Extractor()

# Extract a bunch of audio features http://essentia.upf.edu/documentation/reference/std_Extractor.html
print("\nAnalysing {} audio files...\n".format(len(file_names)))

stats = ["min", "max", "median", "mean", "cov", "kurt", "skew"]

for file in loaded_files:
    current_extractor = extractor(file)

    # Statistical Aggregation http://essentia.upf.edu/documentation/reference/std_PoolAggregator.html
    data_pools_aggregated.append(PoolAggregator(defaultStats = stats)(current_extractor))

# Create result directories if necessary
result_root = os.path.dirname(os.path.realpath(__file__)) + '/../extracted/'
result_directories = [os.path.join(result_root, category) for category in categories]

try:
    for path in result_directories:
        os.makedirs(path)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise

result_paths = map(lambda path: path.replace(file_extension, '') + '.json', file_paths)
result_paths = map(lambda path: path.split('/data')[1], result_paths)

for index, aggregated_pool in enumerate(data_pools_aggregated):
    YamlOutput(filename = result_root + result_paths[index], format = 'json')(aggregated_pool)
