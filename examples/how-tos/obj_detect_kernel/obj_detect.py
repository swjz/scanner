from scannerpy import Database, Job, BulkJob, ColumnType, DeviceType
import os
import sys
import cv2
import math
import skvideo.io
from tqdm import tqdm
from utils import visualization_utils as vis_util
from utils import label_map_util
from multiprocessing import Pool

###############################################################################################
# Assume that DNN model is located in PATH_TO_GRAPH with filename 'frozen_inference_graph.pb' #
###############################################################################################

script_dir = os.path.dirname(os.path.abspath(__file__))
PATH_TO_REPO = script_dir

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(PATH_TO_REPO, 'data', 'mscoco_label_map.pbtxt')

PATH_TO_GRAPH = os.path.join(PATH_TO_REPO, 'ssd_mobilenet_v1_coco_2017_11_17','frozen_inference_graph.pb')

NUM_CLASSES = 90

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
    max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

###############################################################################################

# Intersection Over Union (Area)
def IoU(box1, box2):
    # intersection rectangle (y1, x1, y2, x2)
    y1 = max(box1[0], box2[0])
    x1 = max(box1[1], box2[1])
    y2 = min(box1[2], box2[2])
    x2 = min(box1[3], box2[3])
    area_intersection = (x2 - x1) * (y2 - y1)

    area_box1 = (box1[3] - box1[1]) * (box1[2] - box1[0])
    area_box2 = (box2[3] - box2[1]) * (box2[2] - box2[0])

    area_union = area_box1 + area_box2 - area_intersection

    return area_intersection * 1.0 / area_union

# non-maximum suppression
def nms_single(bundled_data, iou_threshold=0.5):
    bundled_data = bundled_data.reshape(20, 6)
    data_size = len(bundled_data)
    repeated_indices = []
    selected_indices = set(range(data_size))

    [boxes, classes, scores] = np.split(bundled_data, [4, 5], axis=1)

    for i in range(data_size):
        for j in range(i+1, data_size):
            if IoU(boxes[i], boxes[j]) > iou_threshold and classes[i] == classes[j]:
                repeated_indices.append(j)

    repeated_indices = set(repeated_indices)
    selected_indices = list(selected_indices - repeated_indices)

    selected_bundled_data = np.take(bundled_data, selected_indices, axis=0)
    [boxes_np, classes_np, scores_np] = np.split(selected_bundled_data, [4, 5], axis=1)

    return [boxes_np, classes_np, scores_np]

# tried to use multiprocessing module to scale,
# but doesn't work well because of high overhead cost
def nms_bulk(bundled_data_list):
    print("Working on non-maximum suppression...")
    bundled_np_list = [nms_single(bundled_data) for bundled_data in tqdm(bundled_data_list)]
    # bundled_np_list = map(nms_single, tqdm(bundled_data_list))
    print("Finished non-maximum suppression!")
    return bundled_np_list

# This method returns whether two boxes are close enough.
# If two boxes from two neighboring frames are considered
# close enough, they are refered as the same object.
def neighbor_boxes(box1, box2, threshold=0.1):
    if math.abs(box1[0] - box2[0]) > threshold:
        return False
    if math.abs(box1[1] - box2[1]) > threshold:
        return False
    if math.abs(box1[2] - box2[2]) > threshold:
        return False
    if math.abs(box1[3] - box2[3]) > threshold:
        return False
    return True

def smooth_box(bundled_np_list, min_score_thresh=0.5):
    print("Working on making boxes smooth...")
    
    for i, now in enumerate(tqdm(bundled_np_list)):

        # Ignore the first and last frame
        if i == 0:
            before = None
            continue
        else:
            before = bundled_np_list[i - 1]

        if i == len(bundled_np_list) - 1:
            after = None
            continue
        else:
            after = bundled_np_list[i + 1]

        [boxes_after, classes_after, scores_after] = after
        [boxes_before, classes_before, scores_before] = before
        [boxes_now, classes_now, scores_now] = now

        for j, [box_now, class_now, score_now] in enumerate(zip(boxes_now, classes_now, scores_now)):

            # Assume that the boxes list is already sorted
            if score_now < min_score_thresh:
                break

            confirmed_box_after = None
            confirmed_box_before = None

            for k, [box_after, class_after, score_after] in enumerate(zip(boxes_after, classes_after, scores_after)):
                # if neighbor_boxes(box_now, box_after):
                if IoU(box_now, box_after) > 0.3 and class_now == class_after and score_after > min_score_thresh - 0.1:
                    confirmed_box_after = box_after
                    if score_after < min_score_thresh:
                        scores_after[k] = score_now
                    break

            for k, [box_before, class_before, score_before] in enumerate(zip(boxes_before, classes_before, scores_before)):
                if IoU(box_now, box_before) > 0.3 and class_now == class_before:
                    confirmed_box_before = box_before
                    break

            if confirmed_box_before is not None and confirmed_box_after is not None:
                box_now += box_now
                box_now += confirmed_box_before
                box_now += confirmed_box_after
                box_now /= 4.0
            elif confirmed_box_before is not None:
                box_now += confirmed_box_before
                box_now /= 2.0
            elif confirmed_box_after is not None:
                box_now += confirmed_box_after
                box_now /= 2.0

            boxes_now[j] = box_now

        bundled_np_list[i] = [boxes_now, classes_now, scores_now]
        bundled_np_list[i+1] = [boxes_after, classes_after, scores_after]
        bundled_np_list[i-1] = [boxes_before, classes_before, scores_before]

    print("Finished making boxes smooth!")
    return bundled_np_list

def draw_boxes(frame, bundled_np, min_score_thresh=0.5):
    [boxes, classes, scores] = bundled_np
    vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=min_score_thresh)
    return frame

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Usage: {:s} path/to/your/video/file.mp4'.format(sys.argv[0]))
        sys.exit(1)

    movie_path = sys.argv[1]
    print('Detecting objects in movie {}'.format(movie_path))
    movie_name = os.path.splitext(os.path.basename(movie_path))[0]

    bundled_data_list = []
    sample_stride = 1

    with Database() as db:
        [input_table], failed = db.ingest_videos([('example', movie_path)], force=True)
        print(db.summarize())
        db.register_op('ObjDetect', [('frame', ColumnType.Video)], ['bundled_data'])
        kernel_path = script_dir + '/obj_detect_kernel.py'
        db.register_python_kernel('ObjDetect', DeviceType.CPU, kernel_path)
        frame = db.ops.FrameInput()
        strided_frame = frame.sample()

        # Call the newly created object detect op
        objdet_frame = db.ops.ObjDetect(frame = strided_frame)

        output_op = db.ops.Output(columns=[objdet_frame])
        job = Job(
            op_args={
                frame: db.table('example').column('frame'),
                strided_frame: db.sampler.strided(sample_stride),
                output_op: 'example_obj_detect',
            }
        )
        bulk_job = BulkJob(output=output_op, jobs=[job])
        [out_table] = db.run(bulk_job, force=True)

        print('Extracting data from Scanner output...')

        # bundled_data_list is a list of bundled_data
        # bundled data format: [box position(x1 y1 x2 y2), box class, box score]
        bundled_data_list = [np.fromstring(box, dtype=np.float32) for (_, box) in tqdm(out_table.column('bundled_data').load())]
        
        print('Successfully extracted data from Scanner output!')

    # video_frames = []
    # video_frames = [f[0] for _, f in tqdm(db.table('example').load(['frame']))]
    videogen = skvideo.io.vreader(movie_path)
    # for frame in videogen:
    #     # Capture frame-by-frame
    #     video_frames.append(frame)

    print('Loaded frames to numpy array')

    # run non-maximum suppression
    bundled_np_list = nms_bulk(bundled_data_list)
    bundled_np_list = smooth_box(bundled_np_list, min_score_thresh=0.5)

    # draw bounding boxes to video and write video to file
    # frame_shape = videogen[0].shape
    # output = cv2.VideoWriter(
    #     movie_name + '_obj_detect.mkv',
    #     cv2.VideoWriter_fourcc(*'X264'),
    #    #cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
    #     24.0,
    #     (frame_shape[1], frame_shape[0]))

    print('Writing frames to {:s}_obj_detect.mp4'.format(movie_name))

    writer = skvideo.io.FFmpegWriter(movie_name + '_obj_detect.mp4')
    for i, frame in enumerate(tqdm(videogen)):
        frame = draw_boxes(frame, bundled_np_list[i//sample_stride], min_score_thresh=0.5)
        frame = frame.astype(np.uint8)
        writer.writeFrame(frame)
        # output.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).astype(np.uint8))
    writer.close()

    # for (frame, bundled_np) in tqdm(zip(video_frames, bundled_np_list)):
    #     frame = draw_boxes(frame, bundled_np, min_score_thresh=0.5)
    #     output.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).astype(np.uint8))

    print('Successfully generated {:s}_obj_detect.mp4'.format(movie_name))
    # output.release()