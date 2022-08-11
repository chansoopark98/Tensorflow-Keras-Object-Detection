"""coex_hand dataset."""
import tensorflow_datasets as tfds
import os
import glob
import natsort
import numpy as np
import tensorflow as tf
import random


_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(coex_hand): BibTeX citation
_CITATION = """
"""
class CoexHand(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for coex_hand dataset."""
  
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(display_detection): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(None, None, 3)),
            'bbox': tfds.features.Tensor(shape=(None, 4), dtype=tf.float32),
            'label': tfds.features.Tensor(shape=(None,), dtype=tf.int64),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=None,
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""

    # 데이터셋 구성은 다음과 같이 설정해야함.
    """
        └── coex_hand.zip 
        ├── train/  
        |   ├── image/ 
        |   └── label/
        └── validation/  # Semantic label.    
            ├── image/ 
            └── label/
    """

    archive_path = '../data_generate/data/coex_hand/coex_hand.zip'
    extracted_path = dl_manager.extract(archive_path)

    return {
        'train': self._generate_examples(img_path=extracted_path/'train/image', label_path=extracted_path/'train/label'),
        'validation': self._generate_examples(img_path=extracted_path/'validation/image', label_path=extracted_path/'validation/label')
    }

  def _generate_examples(self, img_path, label_path):
    img_list = os.path.join(img_path, '*.png')
    label_path = os.path.join(label_path, '*.txt')
  
    
    img_files = glob.glob(img_list)
    img_files = natsort.natsorted(img_files,reverse=True)

    label_files = glob.glob(label_path)
    label_files = natsort.natsorted(label_files,reverse=True)


    for idx in range(len(img_files)):
      label_output_list = []
      bbox_output_list = []

      with open(str(label_files[idx]), "r") as file:
          read_bbox_list = file.readlines()
            
      
      for i in range(len(read_bbox_list)):
        line_label = read_bbox_list[i].replace('\n', '')
        line_label = list(line_label.split(' '))
        

        label = int(line_label[0])
        bbox = line_label[1:5]
        bbox = [float(box) for box in bbox]
        x_min = bbox[0] - (bbox[2] / 2)
        y_min = bbox[1] - (bbox[3] / 2)
        x_max = bbox[0] + (bbox[2] / 2)
        y_max = bbox[1] + (bbox[3] / 2)

        # clamp
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)

        x_max = min(x_max, 1)
        y_max = min(y_max, 1)
        
        voc_box = [y_min, x_min, y_max, x_max]
        label_output_list.append(label)
        bbox_output_list.append(voc_box)

          
      read_bbox_list = np.array(bbox_output_list).astype(np.float32)
      read_label_list = np.array(label_output_list).astype(np.int64)

      yield idx, {
          'image': img_files[idx],
          'bbox' : read_bbox_list,
          'label': read_label_list
      }

