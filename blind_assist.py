import sys
import tensorflow as tf
import cv2
import numpy as np
import time
from utils import label_map_util
from utils import visualization_utils_color as vis_util
from gtts import gTTS
from pydub import AudioSegment
import subprocess
from threading import Thread

# Path to model
PATH_TO_MODEL = './models/face_model/frozen_inference_graph_face.pb'
# correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'
NUM_CLASSES = 2

global GLOBAL_SPEAK, DESCRIPTION
GLOBAL_SPEAK = 0
DESCRIPTION = ''

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


class FaceDector(object):
    def __init__(self, PATH_TO_MODEL):
        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with self.graph.as_default():
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.compat.v1.Session(
                graph=self.graph, config=config)
            self.windowNotSet = True

    def run(self, image):
        """image: bgr image
        return (boxes, scores, classes, num_detections)
        """

        np_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        np_image_expanded = np.expand_dims(np_image, axis=0)
        image_tensor = self.graph.get_tensor_by_name(
            'image_tensor:0')
        boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        scores = self.graph.get_tensor_by_name('detection_scores:0')
        classes = self.graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.graph.get_tensor_by_name('num_detections:0')

        start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: np_image_expanded})
        elapsed_time = time.time() - start_time
        print('inference time: {}'.format(elapsed_time))
        return (boxes, scores, classes, num_detections)


def speed_change(sound, speed=1.0):
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
         "frame_rate": int(sound.frame_rate * speed)
      })
    return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)


def play_music():
    tts = gTTS(DESCRIPTION, lang='en')
    tts.save('./tts.mp3')
    tts = AudioSegment.from_mp3("./tts.mp3")
    tts = speed_change(tts, 1.1)
    tts.export('./tts_export.mp3')
    subprocess.call(["ffplay", "-nodisp", "-autoexit", "./tts_export.mp3"])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Enter two inputs: either videos or camera streams")
        exit(1)

    try:
        camID = int(sys.argv[1])
        camID2 = int(sys.argv[2])
    except:
        camID = sys.argv[1]
        camID2 = sys.argv[2]

    faceModel = FaceDector(PATH_TO_MODEL)

    cap = cv2.VideoCapture(camID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap2 = cv2.VideoCapture(camID2)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    h0 = None
    windowNotSet = True
    counter = 0
    while True:
        counter += 1
        ret1, image1 = cap.read()
        ret2, image2 = cap2.read()
        # if not counter%5==0:
        #     continue

        if ret1 == 0 or ret2 == 0:
            break
        image1 = cv2.resize(image1, None, fx=0.5, fy=0.5)
        image2 = cv2.resize(image2, None, fx=0.5, fy=0.5)
        [h1, w1] = image1.shape[:2]
        [h2, w2] = image2.shape[:2]

        (boxes1, scores1, classes1, num_detections1) = faceModel.run(image1)
        (boxes2, scores2, classes2, num_detections2) = faceModel.run(image2)

        # print(np.squeeze(classes).astype(np.int32))
        speak1, dist1, num1 = vis_util.visualize_boxes_and_labels_on_image_array(
            image1,
            np.squeeze(boxes1),
            np.squeeze(classes1).astype(np.int32),
            np.squeeze(scores1),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=4,
            front_cam=True)

        speak2, dist2, num2 = vis_util.visualize_boxes_and_labels_on_image_array(
            image2,
            np.squeeze(boxes2),
            np.squeeze(classes2).astype(np.int32),
            np.squeeze(scores2),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=4,
            front_cam=False)

        if windowNotSet is True:
            cv2.namedWindow(
                "tensorflow based (%d, %d)" % (w1*2, h1), cv2.WINDOW_NORMAL)
            windowNotSet = False

        image = np.concatenate((image1, image2), axis=1)
        # h, w = image.shape[:2]
        cv2.imshow("tensorflow based (%d, %d)" % (w1*2, h1), image)

        if (time.time()-GLOBAL_SPEAK > 5):
            if speak1 and not speak2:
                if num1 == 1:
                    DESCRIPTION = f'person in front less than {round(dist1)} feet'
                else:
                    DESCRIPTION = f'2 or more people in front less than {round(dist1)} feet'
                GLOBAL_SPEAK = time.time()
                music_thread = Thread(target=play_music)
                music_thread.start()
            elif speak2 and not speak1:
                if num2 == 1:
                    DESCRIPTION = f'person in back less than {round(dist2)} feet'
                else:
                    DESCRIPTION = f'2 or more people in back less than {round(dist2)} feet'
                GLOBAL_SPEAK = time.time()
                music_thread = Thread(target=play_music)
                music_thread.start()
            elif speak1 and speak2:
                GLOBAL_SPEAK = time.time()
                DESCRIPTION = f'few people in front and back within 20 feet'
                music_thread = Thread(target=play_music)
                music_thread.start()

        # if h0 is None:
        #   [h0, w0] = image.shape[:2]
        #   out = cv2.VideoWriter("./test_out.avi", 0, 25.0, (w0, h0))
        # else:
        #   out.write(image)
        # cv2.imwrite('./front60.png', image)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q') or k == 27:
            break

    cap.release()
    cap2.release()
    # out.release()

# a = {(0.27993708848953247, 0.2853538990020752, 0.8274571299552917, 0.5723817348480225): 1}
# len(a)
#
# from pydub import AudioSegment
# from threading import Thread
#
# from playsound import playsound
# import subprocess
# from gtts import gTTS
#
# def speed_change(sound, speed=1.0):
#     # Manually override the frame_rate. This tells the computer how many
#     # samples to play per second
#     sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
#          "frame_rate": int(sound.frame_rate * speed)
#       })
#      # convert the sound with altered frame rate to a standard frame rate
#      # so that regular playback programs will work right. They often only
#      # know how to play audio at standard frame rate (like 44.1k)
#     return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)
#
#
# def play_music():
#     description = 'person on left'
#     tts = gTTS(description, lang='en')
#     tts.save('./tts.mp3')
#     tts = AudioSegment.from_mp3("./tts.mp3")
#     subprocess.call(["ffplay", "-nodisp", "-autoexit", "./tts.mp3"])
#     # playsound('./tts.mp3')
#
# music_thread = Thread(target=play_music)
# music_thread.start()
#
# description = 'person on left'
# tts = gTTS(description, lang='en')
# tts.save('./tts.mp3')
# tts = AudioSegment.from_mp3("./tts.mp3")
# tts = speed_change(tts, 1.5)
# tts.export('./tts_export.mp3')
# subprocess.call(["ffplay", "-nodisp", "-autoexit", "./tts_export.mp3"])
