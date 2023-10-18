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



runner = None
# if you don't want to see a camera preview, set this to False
show_camera = True
if (sys.platform == 'linux' and not os.environ.get('DISPLAY')):
    show_camera = False


def akida_model_inference(model, processed_features):

    scaling_factor = 15 / np.max(processed_features)
    # Convert to uint8
    processed_features_uint8 = np.uint8(processed_features * scaling_factor)

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

    model = args[0]

    dir_path = os.path.dirname(os.path.realpath(__file__))
    modelfile = os.path.join(dir_path, model)

    print('MODEL: ' + modelfile)

    akida_model = akida.Model(modelfile)
    #processed features from https://studio.edgeimpulse.com/studio/209276/dsp/organization/59

    #ac
    #processed_features = np.array([0.4957, 0.6186, 0.7006, 0.4793, 0.6391, 0.6350, 0.4793, 0.5859, 0.6350, 0.4957, 0.6678, 0.2991, 0.5039, 0.5490, 0.7538, 0.6596, 0.7210, 0.7006, 0.6022, 0.4998, 0.7538, 0.8399, 0.7825, 0.8030, 0.4916, 0.3359, 0.6350, 0.9791, 0.8276, 0.7006, 0.3400, 0.2827, 0.6350, 0.9832, 0.6637, 0.2458, 0.9791, 0.4916, 0.1966, 0.9505, 0.7210, 0.1966, 0.9341, 0.3687, 0.0410, 1.0365, 0.6883, 0.0574, 1.0119, 0.4220, 0.0205])
    #light
    #processed_features = np.array([0.4752, 0.6678, 0.5695, 0.4384, 0.6965, 0.5695, 0.4507, 0.6309, 0.2458, 0.4629, 0.7702, 0.7538, 0.4670, 0.6186, 0.2458, 0.6760, 0.8685, 0.7538, 0.5818, 0.6063, 0.8440, 0.9751, 0.9546, 0.5695, 0.4343, 0.4548, 0.7538, 0.2376, 0.3974, 0.1966, 0.2417, 0.3933, 0.8440, 0.9996, 0.7538, 0.0942, 1.0160, 0.5777, 0.1229, 0.9996, 0.8112, 0.0205, 0.9382, 0.3196, 0.0410, 0.9955, 0.5408, 0.0410, 0.2171, 0.3892, 0.1557])
    #other
    #processed_features = np.array([0.4834, 0.5367, 0.6350, 0.4670, 0.5613, 0.6350, 0.4670, 0.5080, 0.8030, 0.4957, 0.5981, 0.7538, 0.4834, 0.4793, 0.4998, 0.6514, 0.6678, 0.8440, 0.6350, 0.4220, 0.7538, 0.8276, 0.7743, 0.5695, 0.8399, 0.3564, 0.5695, 0.9382, 0.8480, 0.7006, 0.6760, 0.3073, 0.4302, 0.9751, 0.6268, 0.7006, 0.9751, 0.4466, 0.6350, 0.9955, 0.6965, 0.0369, 1.0037, 0.4179, 0.0082, 0.9873, 0.8399, 0.0737, 0.6719, 0.2991, 0.0737])
    #tv
    processed_features = np.array([0.5039, 0.6432, 0.4302, 0.4834, 0.6719, 0.5695, 0.4793, 0.6145, 0.5695, 0.5039, 0.7210, 0.4998, 0.4998, 0.6022, 0.6350, 0.6678, 0.7702, 0.6350, 0.6104, 0.5654, 0.7006, 0.8685, 0.8112, 0.8440, 0.6309, 0.3933, 0.4998, 0.9628, 0.6678, 0.4998, 0.5981, 0.2212, 0.7006, 0.9628, 0.6924, 0.1557, 0.9382, 0.5572, 0.5695, 0.9873, 0.6432, 0.0205, 0.9914, 0.4711, 0.1966, 0.9914, 0.5859, 0.0410, 0.9873, 0.5613, 0.0410])
    

    predictions = akida_model_inference(akida_model, processed_features)

    np.set_printoptions(suppress=True, floatmode='fixed', precision=6)
    softmaxed_pred = scipy.special.softmax(predictions)
    print(softmaxed_pred)

'''
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
        camera.release()
    else:
        raise Exception("Couldn't initialize selected camera.")

    next_frame = 0 # limit to ~10 fps here

    for res, img in runner.classifier(videoCaptureDeviceId):
        if (next_frame > now()):
            time.sleep((next_frame - now()) / 1000)

        # print('classification runner response', res)

        if "classification" in res["result"].keys():
            print('Result (%d ms.) ' % (res['timing']['dsp'] + res['timing']['classification']), end='')
            for label in labels:
                score = res['result']['classification'][label]
                print('%s: %.2f\t' % (label, score), end='')
            print('', flush=True)

        elif "bounding_boxes" in res["result"].keys():
            print('Found %d bounding boxes (%d ms.)' % (len(res["result"]["bounding_boxes"]), res['timing']['dsp'] + res['timing']['classification']))
            for bb in res["result"]["bounding_boxes"]:
                print('\t%s (%.2f): x=%d y=%d w=%d h=%d' % (bb['label'], bb['value'], bb['x'], bb['y'], bb['width'], bb['height']))
                img = cv2.rectangle(img, (bb['x'], bb['y']), (bb['x'] + bb['width'], bb['y'] + bb['height']), (255, 0, 0), 1)

        if (show_camera):
            cv2.imshow('edgeimpulse', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) == ord('q'):
                break

            next_frame = now() + 100
'''


if __name__ == "__main__":
   main(sys.argv[1:])
