import os
from pandas import wide_to_long
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
from tensorflow.keras.models import load_model
import cv2 
import numpy as np
import glob
from matplotlib import pyplot as plt
from utils import CvFpsCalc



CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'




curr_file_dir = os.path.dirname(os.path.abspath(__file__))

paths = {
    'WORKSPACE_PATH': os.path.join(curr_file_dir,'Tensorflow', 'workspace'),
    'MAHJONG_PATH': os.path.join(curr_file_dir,'Tensorflow','mahjong'),
    'SCRIPTS_PATH': os.path.join(curr_file_dir,'Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join(curr_file_dir,'Tensorflow','models'),
    'IMAGE_PATH': os.path.join(curr_file_dir,'custom_eval_folder'),
    'MODEL_PATH': os.path.join(curr_file_dir,'Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join(curr_file_dir,'Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join(curr_file_dir,'Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join(curr_file_dir,'Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join(curr_file_dir,'Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join(curr_file_dir,'Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join(curr_file_dir,'Tensorflow','protoc')
    
 }

files = {
    'PIPELINE_CONFIG':os.path.join(curr_file_dir,'Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(curr_file_dir, paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(curr_file_dir, paths['MAHJONG_PATH'], LABEL_MAP_NAME)
}

tiles_map = {1: 'bam-1', 2: 'bam-2', 3: 'bam-3', 4: 'bam-4', 5: 'bam-5', 6: 'bam-6', 7: 'bam-7', 8: 'bam-8', 9: 'bam-9', 10: 'char-1', 11: 'char-2', 12: 'char-3', 13: 'char-4', 14: 'char-5', 15: 'char-6', 16: 'char-7', 17: 'char-8', 18: 'char-9', 19: 'dots-1', 20: 'dots-2', 21: 'dots-3', 22: 'dots-4', 23: 'dots-5', 24: 'dots-6', 25: 'dots-7', 26: 'dots-8', 27: 'dots-9', 28: 'h-east', 29: 'h-green', 30: 'h-north', 31: 'h-red', 32: 'h-south', 33: 'h-west', 34: 'h-white'}

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
print(files['PIPELINE_CONFIG'])

# Restore Mahjong Detection model checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-3')).expect_partial()

# Load Mahjong CNN Recognition model
cnn_mahjong_recognition_model = load_model(os.path.join(paths['MODEL_PATH'], 'CNN_Mahjong_recognition_model', 'mahjongclassifier3.h5'))

cap = cv2.VideoCapture(0)
camera_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
camera_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if cap is None or not cap.isOpened():
       print('Warning: unable to open video source:', 0)
print(cap)
player_hand = []

# detection fuction for mahjon object detection
@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

# helper function to process bounding box detected by mahjong detection model
def processBoxImage(x1, y1, x2, y2, img):
    resize = tf.image.resize(img[y1:y2, x1:x2], (256,256))
    grayscale_img = tf.image.rgb_to_grayscale(resize)
    yhat = cnn_mahjong_recognition_model(np.expand_dims(grayscale_img/255, 0), training=False).numpy()
    result = yhat.tolist()
    prediction_id = result[0].index(max(result[0])) + 1
    prediction_name = tiles_map[prediction_id]
    
    return prediction_id, prediction_name, round(max(result[0]) * 100)

# FPS Measurement ########################################################
cvFpsCalc = CvFpsCalc(buffer_len=10)

# Mahjong AI #############################################################
def tile_combo(tile_i, tile_j, tile_k):
    # return true if the three tiles form a combo
    # check out of bounds
    if tile_i < 0 or tile_i > 34 or tile_j < 0 or tile_j > 34 or tile_k < 0 or tile_k > 34:
        return False
    # check if they are all the same
    if tile_i == tile_j and tile_j == tile_k:
        return 1
    # check if tiles are consecutive
    elif tile_i + 1 == tile_j and tile_j + 1 == tile_k:
        # check if tiles are in range 1 - 9, or 10 - 18, or 19 - 27
        if tile_k < 10 or (tile_i >= 10 and tile_k < 19) or (tile_i >= 19 and tile_k < 28):
            return 1
    return 0

def get_max_combo(tiles):
    memo = []
    sorted_Tiles = sorted(tiles)
    for i in range(len(tiles)):
        if i < 2:
            memo.append(0)
        else:
            memo.append(max(0, memo[i - 1], memo[i - 3] + tile_combo(sorted_Tiles[i - 2], sorted_Tiles[i - 1], sorted_Tiles[i])))
    # print(memo)
    result = []
    # iterate through memo, and add three tiles to result when memo[i] > memo[i - 1]
    rest = sorted_Tiles.copy()
    for i in range(len(memo)):
        if memo[i] > memo[i - 1]:
            result.append([sorted_Tiles[i - 2], sorted_Tiles[i - 1], sorted_Tiles[i]])
            rest.remove(sorted_Tiles[i - 2])
            rest.remove(sorted_Tiles[i - 1])
            rest.remove(sorted_Tiles[i])
    result.append(rest)
    # print(result)
    return result

def display_max_combo(tiles):
    max_combo = get_max_combo(tiles)
    result = max_combo.copy()
    for combo in range(len(max_combo)):
        for tiles in range(len(max_combo[combo])):
            result[combo][tiles] = tiles_map[max_combo[combo][tiles]]
    return result

# assume that tiles are sorted
def get_pairs(tiles):
    pairs = []
    for i in range(len(tiles) - 1):
        if tiles[i] == tiles[i + 1]:
            pairs.append([tiles[i], tiles[i + 1]])
    return pairs
    

def display_max_combo_img(tiles, frame):
    max_combo = get_max_combo(tiles)
    horizontal_offset = 15
    vertical_offset = camera_height - 100
    mahjong_display_path = os.path.join(curr_file_dir, 'utils', 'mahjong_display')
    for combo in range(len(max_combo)):
        for tiles in range(len(max_combo[combo])):
            tile_name = tiles_map[max_combo[combo][tiles]]
            tile_img = cv2.imread(os.path.join(mahjong_display_path, tile_name + '.jpg'))
            frame[vertical_offset:vertical_offset + tile_img.shape[0], horizontal_offset:horizontal_offset + tile_img.shape[1]] = tile_img
            horizontal_offset += tile_img.shape[1]
        # add a space between each combo
        horizontal_offset += 10
    rest = max_combo[-1]
    hu = []
    game_finsihed = False
    
    if len(rest) == 1:
        # check hu condition 让俺看看这一手能不能胡
        hu.append(rest[0])
    elif len(rest) == 2 and rest[0] == rest[1]:
        game_finsihed = True
    elif len(rest) == 4:
        # 四个里面其中有两个对子，另外两个是是顺子，那么这一手能胡
        # check if there is a pair in rest
        pairs = get_pairs(rest)
        remain_rest = rest.copy()
        # remove pairs from rest
        for pair in pairs:
            remain_rest.remove(pair[0])
            remain_rest.remove(pair[1])
        if len(remain_rest) == 0 and len(pairs) == 2:
            # all tiles are pairs, 两个对子，等任意一个碰 就可以胡
            hu.append(pairs[0][0])
            hu.append(pairs[1][0])
        elif len(remain_rest) == 2 and len(pairs) == 1:
            # check if remain_rest can form a combo with any extra tile
            # 看看剩下的两张牌能不能组成一个顺子
            if tile_combo(remain_rest[0], remain_rest[1], remain_rest[1] + 1):
                hu.append(remain_rest[1] + 1)
            if tile_combo(remain_rest[0], remain_rest[0] + 1, remain_rest[1]):
                hu.append(remain_rest[0] + 1)
            if tile_combo(remain_rest[0] - 1, remain_rest[0], remain_rest[1]):
                hu.append(remain_rest[0] - 1)
    hu_x_offset = camera_width - 20
    hu_y_offset = 100
    # display hu tiles
    for hu_tile in hu:
        tile_name = tiles_map[hu_tile]
        tile_img = cv2.imread(os.path.join(mahjong_display_path, tile_name + '.jpg'))
        frame[hu_y_offset:hu_y_offset + tile_img.shape[0], hu_x_offset - tile_img.shape[1]:hu_x_offset] = tile_img
        hu_x_offset -= tile_img.shape[1]
    # draw rectangle around hu tiles
    if len(hu) > 0:
        tile = tiles_map[hu[0]]
        tile_img = cv2.imread(os.path.join(mahjong_display_path, tile + '.jpg'))
        cv2.rectangle(frame, (hu_x_offset  - 10, hu_y_offset - 10), (camera_width - 10, hu_y_offset + tile_img.shape[0] + 10), (255, 255, 255), 1)
            
    return frame



while cap.isOpened(): 
    fps = cvFpsCalc.get()
    ret, frame = cap.read()
    image_np = np.array(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    curr_img = image_np.copy()
    label_id_offset = 1
    boxes = detections['detection_boxes']
    scores = detections['detection_scores']
    min_score_thresh = 0.8
    player_hand_detection = []
    for box_i in range(boxes.shape[0]):
        if scores is None or scores[box_i] > min_score_thresh:
            ymin, xmin, ymax, xmax = boxes[box_i]
            y1 = int(ymin * camera_height)
            y2 = int(ymax * camera_height)
            x1 = int(xmin * camera_width)
            x2 = int(xmax * camera_width)
            # cv2.imwrite(os.path.join(paths['IMAGE_PATH'], 'results', '{}-{}-{}-{}.jpg'.format(x1, y1, x2, y2)), curr_img[y1:y2, x1:x2])
            prediction_id, prediction_name, score = processBoxImage(x1, y1, x2, y2, curr_img)
            cv2.rectangle(curr_img, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.putText(curr_img, "{}".format(prediction_name), (x1,y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (255,0,0), 1)
            cv2.putText(curr_img, "{}%".format(score), (x1,y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (255,0,0), 1)
            player_hand_detection.append(prediction_id)
    
    # previously detected tiles different from new detected tiles
    if player_hand != player_hand_detection:
        player_hand = player_hand_detection

    # get the best combinations of tiles
    curr_img = display_max_combo_img(player_hand, curr_img)
    
    cv2.putText(curr_img, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 0), 4, cv2.LINE_AA)
    # cv2.putText(curr_img, "{}".format(best_combos), (20, camera_height - 30), cv2.FONT_HERSHEY_SIMPLEX,
    # 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow('object detection',  curr_img)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()