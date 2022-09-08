"""voc_zero dataset."""
import tensorflow_datasets as tfds
import os
import glob
import natsort
import numpy as np
import tensorflow as tf
import random


# TODO(voc_zero): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(voc_zero): BibTeX citation
_CITATION = """
"""


class VocZero(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for voc_zero dataset."""
  # MANUAL_DOWNLOAD_INSTRUCTIONS = '/home/park/tensorflow_datasets/'
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
    # TODO(cornell_grasp): Downloads the data and defines the splits
    # archive_path = dl_manager.manual_dir / 'display_detection.zip'
    archive_path = '../datasets/voc_zero_raw/voc_zero.zip'
    extracted_path = dl_manager.extract(archive_path)

    # TODO(cornell_grasp): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(img_path=extracted_path/'rgb',
                                         bbox_path=extracted_path/'bbox',
                                         label_path=extracted_path/'label')
    }

  def _generate_examples(self, img_path, bbox_path, label_path):
    img_list = os.path.join(img_path, '*.png')
    bbox_list = os.path.join(bbox_path, '*.txt')
    label_list = os.path.join(label_path, '*.txt')
    
    img_files = glob.glob(img_list)
    img_files = natsort.natsorted(img_files,reverse=True)

    bbox_files = glob.glob(bbox_list)
    bbox_files = natsort.natsorted(bbox_files,reverse=True)

    label_files = glob.glob(label_list)
    label_files = natsort.natsorted(label_files,reverse=True)

    
    for idx in range(len(img_files)):

      with open(str(bbox_files[idx]), "r") as file:
          read_bbox_list = file.readlines()
            
      with open(str(label_files[idx]), "r") as file:
          read_label_list = file.readlines()

      try:  
        for i in range(len(read_bbox_list)):
            read_bbox_list[i] = read_bbox_list[i].replace('\n', '')
            read_label_list[i] = read_label_list[i].replace('\n', '')

            read_label_list[i] = int(read_label_list[i])

            # bbox_list[i] # str, x1, y1, x2, y2
            read_bbox_list[i] = read_bbox_list[i].replace('[', '')
            read_bbox_list[i] = read_bbox_list[i].replace(']', '')
            bbox_batch = read_bbox_list[i].split(',')

            batch_box_out = []
            for j in range(len(bbox_batch)):
                bbox_batch[j] = bbox_batch[j].replace(' ', '')    
                batch_box_out.append(float(bbox_batch[j]))
            
            
            read_bbox_list[i] = batch_box_out

          
        read_bbox_list = np.array(read_bbox_list).astype(np.float32)
        read_label_list = np.array(read_label_list).astype(np.int64)

      except Exception as e:
        print('error', e)
        print('bbox_files', bbox_files[idx])
        print('label_files', label_files[idx])
        print('read_bbox_list', read_bbox_list)
        print('read_label_list', read_label_list)
      # read_bbox_list = np.squeeze(read_bbox_list, 0)
      # read_label_list = np.squeeze(read_label_list, 0)

      yield idx, {
          'image': img_files[idx],
          'bbox' : read_bbox_list,
          'label': read_label_list
      }

