from __future__ import print_function

import tensorflow as tf
import numpy as np
import cv2
import six.moves.urllib as urllib
import os
import sys
import time
import tarfile
import zipfile
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
BASE_PATH = '/home/jonas/.virtualenvs/object_detection_py3/lib/python3.5/site-packages/tensorflow/models/object_detection'
sys.path.append( BASE_PATH )

from utils import label_map_util
from utils import visualization_utils as vis_util

# Helper functions
def check_download( model_name ):
    """Check if model should be downloaded"""
    if os.path.isdir( model_name ):
        print( 'Model already downloaded, using existing version' )
    else:
        print( 'Requested model not found, attempting to download' )
        download_pretrained_model( model_name )

def load_image_into_numpy( image ):
    """Return an image as a numpy array."""
    ( image_width, image_height ) = image.size
    return np.array( image.getdata() ).reshape( image_height, image_width, 3 ).astype( np.uint8 )

def download_pretrained_model( model_name ):
    """Download a new pretrained network."""
    MODEL_TAR_FILE = model_name + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    try:
        opener = urllib.request.URLopener()
        opener.retrieve( DOWNLOAD_BASE + MODEL_TAR_FILE, MODEL_TAR_FILE )
        tar_library = tarfile.open( MODEL_TAR_FILE )
        for file in tar_library.getmembers():
            file_name = os.path.basename( file.name )
            if 'frozen_inference_graph.pb' in file_name:
                tar_library.extract( file, os.getcwd() )
        os.remove( MODEL_TAR_FILE )
        print( 'Download successful' )
    except:
        sys.exit( 'Error, unable to find download link. Did you type it correctly?' )

def load_frozen_graph( graph, ckpt_path ):
    """Load a pretrained tensorflow graph from checkpoint.
    
    Arguments:
        graph -- A tensorflow graph instance
        ckpt_path -- Path to directory of pretrained model state checkpoint
    """
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile( ckpt_path, 'rb' ) as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString( serialized_graph )
            tf.import_graph_def( od_graph_def, name = '' )


def main( args ):
    """Detect objects in webcamera feed."""
    model = str( args.network ) if args.network is not None else 'ssd_mobilenet_v1'
    total_frames = args.frames if args.frames is not None else None
    do_visualize = args.visualize
    doot = args.doot

    print( 'Using pretrained network model ' + model )
    MODEL_NAME = model + '_coco_11_06_2017'
    check_download( MODEL_NAME )
        
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = BASE_PATH + '/data/mscoco_label_map.pbtxt'

    N_CLASSES = 90
    SCORE_THRESHOLD = .5 # Threshold used by bounding-box plot function
    PRINT_INTERVAL = 10 # Number of frames between each console output
    ALERT_DURATION = 0.2  # Seconds
    ALERT_FREQUENCY = 800  # Hz

    cam = cv2.VideoCapture( 0 )
    print( cam.isOpened() )
    if cam.isOpened() == False:
        print( 'Failed to connect to camera, aborting' )
        exit()

    detection_graph = tf.Graph()
    load_frozen_graph( detection_graph, PATH_TO_CKPT )

    label_map = label_map_util.load_labelmap( PATH_TO_LABELS )
    categories = label_map_util.convert_label_map_to_categories( label_map, max_num_classes = N_CLASSES, 
                                                                            use_display_name = True )
    category_index = label_map_util.create_category_index( categories )
    
    n_frames = 0
    with detection_graph.as_default():
        with tf.Session( graph = detection_graph ) as sess:
            running = True
            start_time = time.time()

            while running:
                n_frames += 1
                _, image = cam.read()
                image_expanded = np.expand_dims( image, axis = 0 )

                image_tensor = detection_graph.get_tensor_by_name( 'image_tensor:0' )
                detection_boxes = detection_graph.get_tensor_by_name( 'detection_boxes:0' )
                detection_scores = detection_graph.get_tensor_by_name( 'detection_scores:0' )
                detection_classes = detection_graph.get_tensor_by_name( 'detection_classes:0' )
                num_detections = detection_graph.get_tensor_by_name( 'num_detections:0' )

                ( boxes, scores, classes, n_detections ) = sess.run( [detection_boxes, detection_scores,
                                                                      detection_classes, num_detections],
                                                                      feed_dict = {image_tensor: image_expanded} )

                if do_visualize:
                    vis_util.visualize_boxes_and_labels_on_image_array( image, np.squeeze(boxes),
                                                                        np.squeeze(classes).astype(np.int32),
                                                                        np.squeeze(scores), category_index,
                                                                        use_normalized_coordinates = True,
                                                                        line_thickness = 5 )
                    cv2.imshow( 'Object detection', cv2.resize(image, (800, 600)) )

                
                if n_frames % PRINT_INTERVAL == 0:
                    num_detections = sum( sum(scores > SCORE_THRESHOLD) )
                    print( 'Number of objects detected:' )
                    print( num_detections )

                    if num_detections > 0:
                        primary_object = category_index.values()[ int(classes[0, 0] - 1) ].get( 'name' )
                        print( 'Found objects and scores:' )

                        for i in range( num_detections ):
                            next_class = classes[ 0, i ]
                            object_score = scores[ 0, i ]
                            next_object = category_index.get( next_class ).get( 'name' )
                            print( next_object, ': ', object_score )

                        if primary_object == 'person' and doot:
                            os.system( 'play --no-show-progress --null --channels 1 synth %s sine %f' % (ALERT_DURATION, ALERT_FREQUENCY) )
                    
                    print( '-'*27)
                
                if cv2.waitKey( 1 ) & 0xFF == ord( 'q' ):
                    running = False
                    elapsed_time = time.time() - start_time
                elif total_frames is not None and n_frames == total_frames:
                    running = False
                    elapsed_time = time.time() - start_time


    cam.release()
    cv2.destroyAllWindows()
    
    fps = n_frames / elapsed_time
    print( 'Average FPS: ', round(fps, 2) )
    
if __name__ == '__main__':
    """Detect objects in web camera video feed.
    
    Arguments:
        'network' (string), should be one of
            ssd_mobilenet_v1 -- FAST, lower performance
            ssd_inception_v2 -- FAST, lower performance (Default network)
            rfcn_resnet101 -- MEDIUM speed, better performance
            faster_rcnn_resnet101 -- MEDIUM speed, better performance
            faster_rcnn_inception_resnet_v2_atrous -- SLOW, best performance
        'frames' (integer), number of frames > 0
        'visualize', display video feed and bounding boxes
        'doot', play a doot when  a human is detected
    """
    parser = argparse.ArgumentParser()
    parser.add_argument( '-n', '--network', help = 'Architecture of classifier network (default ssd_inception_v2)' )
    parser.add_argument( '-f', '--frames', type = int, help = 'The number of frames to capture (default inf)' )
    parser.add_argument( '-v', '--visualize', action = 'store_false', default = True, help = 'Visualize camera feed and bounding boxes (default True)' ) 
    parser.add_argument( '-d', '--doot', action = 'store_true', default = False, help = 'Doot at humans (default False)' )

    args = parser.parse_args()
    print( '-'*39,'\nRunning object detection through webcam' )
    print( '-'*39 )
    main( args )