# Mostly taken from: https://github.com/tensorflow/models/blob/master/object_detection/object_detection_tutorial.ipynb

import numpy as np
import tensorflow as tf
import cv2
import os
from scannerpy.stdlib import kernel
from utils import visualization_utils as vis_util
from utils import label_map_util
import six.moves.urllib as urllib

script_dir = os.path.dirname(os.path.abspath(__file__))
PATH_TO_REPO = script_dir

# # What model to download.
# MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# # List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join(PATH_TO_REPO, 'data', 'mscoco_label_map.pbtxt')

PATH_TO_GRAPH = os.path.join(PATH_TO_REPO, 'ssd_mobilenet_v1_coco_2017_11_17', 'frozen_inference_graph.pb')

# NUM_CLASSES = 90

# label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
# category_index = label_map_util.create_category_index(categories)

class ObjDetectKernel(kernel.TensorFlowKernel):
    def build_graph(self):
        dnn = tf.Graph()
        with dnn.as_default():
            od_graph_def = tf.GraphDef()
            print("Loading DNN model...")
            with tf.gfile.GFile(PATH_TO_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            print("Successfully loaded DNN model!")
        return dnn

    # Evaluate object detection DNN model on a frame
    # Return bounding box position, class and score
    def execute(self, cols):
        image = cols[0]
        image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        scores = self.graph.get_tensor_by_name('detection_scores:0')
        classes = self.graph.get_tensor_by_name('detection_classes:0')
        with self.graph.as_default():
            (boxes, scores, classes) = self.sess.run(
                [boxes, scores, classes],
                feed_dict={image_tensor: np.expand_dims(image, axis=0)})
            
            # bundled data format: [box position(x1 y1 x2 y2), box class, box score]
            bundled_data = np.concatenate((boxes.reshape(100,4), classes.reshape(100,1), scores.reshape(100,1)), 1)[:20]
            
            return [bundled_data.tobytes()]

KERNEL = ObjDetectKernel
