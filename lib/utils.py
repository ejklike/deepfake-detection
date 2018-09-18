import os
import random
random.seed(1)

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return directory


def get_image_paths_and_labels(dir, test_ratio=0.2): # ./data/train/1_t
    image_paths_labels = []
    depth_criteria = len(dir.split('/')) + 1
    for subdir, dirs, files in os.walk(dir):
        # print(subdir, end=' ->' + subdir[-1] + ', ')
        hierarchy = len(subdir.split('/'))
        if hierarchy == depth_criteria: # depth == 1
            try:
                label = {
                    'f':1,
                    't':0
                }[subdir[-1]]
            except:
                raise ValueError(subdir, ': directory class is not specified.')
            # print('label=', label, end=', ')
            files = [os.path.join(subdir, f) for f in files]
            image_paths_labels += zip(files, [label] * len(files))
            # print(image_paths_labels[-3:])
            # print('#file=', len(files))
        else:
            # print('pass')
            continue
    
    random.shuffle(image_paths_labels)
    n_test = int(len(image_paths_labels) * test_ratio)
    return image_paths_labels[:-n_test], image_paths_labels[-n_test:]