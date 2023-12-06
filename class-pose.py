#!/usr/bin/env python
import cv2
import os
import sys, getopt
import signal
import time

import akida
import scipy
from akida import devices
import numpy as np
from scipy import special


#movenet stuff
import tensorflow as tf
import numpy as np

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

# Some modules to display an animation using imageio.
import imageio
from IPython.display import HTML, display


#@title Helper functions for visualization

url = 'http://10.0.0.173:8123/api/services/google_assistant_sdk/send_text_command'
auth = 'Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiIwNTA2N2M1YjVkMmY0NTIxOGQ2ZjM1ZDZlMmI3OGEwNCIsImlhdCI6MTYyNDkyODA2NCwiZXhwIjoxOTQwMjg4MDY0fQ.jRSQWYe3LpkZO_4No_RWnNhWvX73jpoS6_r91-nEjLU'


# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def _keypoints_and_edges_for_display(keypoints_with_scores,
                                     height,
                                     width,
                                     keypoint_threshold=0.11):
  """Returns high confidence keypoints and edges for visualization.

  Args:
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    height: height of the image in pixels.
    width: width of the image in pixels.
    keypoint_threshold: minimum confidence score for a keypoint to be
      visualized.

  Returns:
    A (keypoints_xy, edges_xy, edge_colors) containing:
      * the coordinates of all keypoints of all detected entities;
      * the coordinates of all skeleton edges of all detected entities;
      * the colors in which the edges should be plotted.
  """
  keypoints_all = []
  keypoint_edges_all = []
  edge_colors = []
  num_instances, _, _, _ = keypoints_with_scores.shape
  for idx in range(num_instances):
    kpts_x = keypoints_with_scores[0, idx, :, 1]
    kpts_y = keypoints_with_scores[0, idx, :, 0]
    kpts_scores = keypoints_with_scores[0, idx, :, 2]
    kpts_absolute_xy = np.stack(
        [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
    kpts_above_thresh_absolute = kpts_absolute_xy[
        kpts_scores > keypoint_threshold, :]
    keypoints_all.append(kpts_above_thresh_absolute)

    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
      if (kpts_scores[edge_pair[0]] > keypoint_threshold and
          kpts_scores[edge_pair[1]] > keypoint_threshold):
        x_start = kpts_absolute_xy[edge_pair[0], 0]
        y_start = kpts_absolute_xy[edge_pair[0], 1]
        x_end = kpts_absolute_xy[edge_pair[1], 0]
        y_end = kpts_absolute_xy[edge_pair[1], 1]
        line_seg = np.array([[x_start, y_start], [x_end, y_end]])
        keypoint_edges_all.append(line_seg)
        edge_colors.append(color)
  if keypoints_all:
    keypoints_xy = np.concatenate(keypoints_all, axis=0)
  else:
    keypoints_xy = np.zeros((0, 17, 2))

  if keypoint_edges_all:
    edges_xy = np.stack(keypoint_edges_all, axis=0)
  else:
    edges_xy = np.zeros((0, 2, 2))
  return keypoints_xy, edges_xy, edge_colors


def draw_prediction_on_image(
    image, keypoints_with_scores, crop_region=None, close_figure=False,
    output_image_height=None):
  """Draws the keypoint predictions on image.

  Args:
    image: A numpy array with shape [height, width, channel] representing the
      pixel values of the input image.
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    crop_region: A dictionary that defines the coordinates of the bounding box
      of the crop region in normalized coordinates (see the init_crop_region
      function below for more detail). If provided, this function will also
      draw the bounding box on the image.
    output_image_height: An integer indicating the height of the output image.
      Note that the image aspect ratio will be the same as the input image.

  Returns:
    A numpy array with shape [out_height, out_width, channel] representing the
    image overlaid with keypoint predictions.
  """
  height, width, channel = image.shape
  aspect_ratio = float(width) / height
  fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
  # To remove the huge white borders
  fig.tight_layout(pad=0)
  ax.margins(0)
  ax.set_yticklabels([])
  ax.set_xticklabels([])
  plt.axis('off')

  im = ax.imshow(image)
  line_segments = LineCollection([], linewidths=(4), linestyle='solid')
  ax.add_collection(line_segments)
  # Turn off tick labels
  scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

  (keypoint_locs, keypoint_edges,
   edge_colors) = _keypoints_and_edges_for_display(
       keypoints_with_scores, height, width)

  line_segments.set_segments(keypoint_edges)
  line_segments.set_color(edge_colors)
  if keypoint_edges.shape[0]:
    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
  if keypoint_locs.shape[0]:
    scat.set_offsets(keypoint_locs)

  if crop_region is not None:
    xmin = max(crop_region['x_min'] * width, 0.0)
    ymin = max(crop_region['y_min'] * height, 0.0)
    rec_width = min(crop_region['x_max'], 0.99) * width - xmin
    rec_height = min(crop_region['y_max'], 0.99) * height - ymin
    rect = patches.Rectangle(
        (xmin,ymin),rec_width,rec_height,
        linewidth=1,edgecolor='b',facecolor='none')
    ax.add_patch(rect)

  fig.canvas.draw()
  image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  image_from_plot = image_from_plot.reshape(
      fig.canvas.get_width_height()[::-1] + (3,))
  plt.close(fig)
  if output_image_height is not None:
    output_image_width = int(output_image_height / height * width)
    image_from_plot = cv2.resize(
        image_from_plot, dsize=(output_image_width, output_image_height),
         interpolation=cv2.INTER_CUBIC)
  return image_from_plot

def to_gif(images, duration):
  """Converts image sequence (4D numpy array) to gif."""
  imageio.mimsave('./animation.gif', images, duration=duration)
  return embed.embed_file('./animation.gif')

def progress(value, max=100):
  return HTML("""
      
          {value}
      
  """.format(value=value, max=max))

runner = None
# if you don't want to see a camera preview, set this to False
show_camera = True
if (sys.platform == 'linux' and not os.environ.get('DISPLAY')):
    show_camera = False


def akida_model_inference(model, processed_features):

    # Convert to uint8
    processed_features_uint8 = np.uint8(processed_features)

    # Reshape to model input shape
    input_shape = (1,) + tuple(model.input_shape)  # Assuming model.input_shape returns (39,)
    inputs = processed_features_uint8.reshape(input_shape)

    # Perform inference
    device = devices()[0]
    model.map(device)
    # model.summary()
    results = model.predict(inputs)
    print(results)
    return results


def now():
    return round(time.time() * 1000)

def get_webcams():
    port_ids = []
    for port in range(5):
        print("Looking for a camera in port %s:" %port)
        camera = cv2.VideoCapture(port)
        if camera.isOpened():
            ret = camera.read()[0]
            if ret:
                backendName =camera.getBackendName()
                w = camera.get(3)
                h = camera.get(4)
                print("Camera %s (%s x %s) found in port %s " %(backendName,h,w, port))
                port_ids.append(port)
            camera.release()
    return port_ids

def sigint_handler(sig, frame):
    print('Interrupted')
    if (runner):
        runner.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)

def help():
    print('python classify.py <path_to_model.fbz> <Camera port ID, only required when more than 1 camera is present>')

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h", ["--help"])
    except getopt.GetoptError:
        help()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            help()
            sys.exit()

    if len(args) == 0:
        help()
        sys.exit(2)

    if len(args)>= 2:
        videoCaptureDeviceId = int(args[1])
    else:
        port_ids = get_webcams()
        if len(port_ids) == 0:
            raise Exception('Cannot find any webcams')
        if len(args)<= 1 and len(port_ids)> 1:
            raise Exception("Multiple cameras found. Add the camera port ID as a second argument to use to this script")
        videoCaptureDeviceId = int(port_ids[0])

    camera = cv2.VideoCapture(videoCaptureDeviceId)
    ret = camera.read()[0]
    if ret:
        backendName = camera.getBackendName()
        w = camera.get(3)
        h = camera.get(4)
        print("Camera %s (%s x %s) in port %s selected." %(backendName,h,w, videoCaptureDeviceId))
    else:
        raise Exception("Couldn't initialize selected camera.")        

    model = args[0]

    dir_path = os.path.dirname(os.path.realpath(__file__))
    modelfile = os.path.join(dir_path, model)

    print('MODEL: ' + modelfile)

    akida_model = akida.Model(modelfile)

    # Initialize the TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    input_size = 192

    def movenet(input_image):
        """Runs detection on an input image.

        Args:
        input_image: A [1, height, width, 3] tensor represents the input image
            pixels. Note that the height/width should already be resized and match the
            expected input resolution of the model before passing into this function.

        Returns:
        A [1, 1, 17, 3] float numpy array representing the predicted keypoint
        coordinates and scores.
        """
        # TF Lite format expects tensor type of uint8.
        input_image = tf.cast(input_image, dtype=tf.uint8)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
        # Invoke inference.
        interpreter.invoke()
        # Get the model prediction.
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
        return keypoints_with_scores    
    
    while True:
      # Load the input image.
      next_frame = 0 # limit to ~10 fps here
      result, image = camera.read()
      cv2.imwrite("image.jpg", image)
      image = tf.io.read_file("image.jpg")
      image = tf.image.decode_jpeg(image)


      # Resize and pad the image to keep the aspect ratio and fit the expected size.
      input_image = tf.expand_dims(image, axis=0)
      input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

      # Run model inference.
      keypoints_with_scores = movenet(input_image)
      scaling_factor = 15
      keypoints_with_scores_scaled = scaling_factor * keypoints_with_scores
      keypoints_with_scores_scaled = keypoints_with_scores_scaled.round()
      keypoints_with_scores_scaled = keypoints_with_scores_scaled.astype(int)    
      keypoints_with_scores_scaled = np.clip(keypoints_with_scores_scaled, 0, 15)
      keypoints_with_scores_flat = keypoints_with_scores_scaled.flatten()
      #print("keypoints with scores = ", keypoints_with_scores)

      # Visualize the predictions with image.
      display_image = tf.expand_dims(image, axis=0)
      display_image = tf.cast(tf.image.resize_with_pad(
          display_image, 1280, 1280), dtype=tf.int32)
      output_overlay = draw_prediction_on_image(
          np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores)

      plt.figure(figsize=(5, 5))
      plt.imshow(output_overlay)
      _ = plt.axis('off')    

      processed_features = keypoints_with_scores_flat

      predictions = akida_model_inference(akida_model, processed_features)

      np.set_printoptions(suppress=True, floatmode='fixed', precision=6)
      softmaxed_pred = scipy.special.softmax(predictions)
      print(softmaxed_pred[0][0][0])
      print("AC: ")
      print(softmaxed_pred[0][0][0][0])
      print(" Light: ")
      print(softmaxed_pred[0][0][0][1])
      print(" Other: ")
      print(softmaxed_pred[0][0][0][2])
      print(" TV: ")
      print(softmaxed_pred[0][0][0][3])

      plt.text(0, 0, np.array2string(softmaxed_pred), fontsize=12)
      plt.savefig('./static/pose.jpg')
      
      #plt.show()

if __name__ == "__main__":
  main(sys.argv[1:])
