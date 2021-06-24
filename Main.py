from FaceClassification import FaceClassification
from FaceDetection import FaceDetection

import getopt,sys

def perform_face_classification():
    
    face_class = FaceClassification()
    face_class.get_names()
    face_class.get_image_paths()
    face_class.recognize_faces()
    face_class.save_to_file()


def perform_face_detection():
    
    face_dect = FaceDetection()
    face_dect.load_data()
    face_dect.start_video_detection()

def main(argv):
    train = False
    try:
      opts, args = getopt.getopt(argv,"h:t:",["train="])
    except getopt.GetoptError:
        print ('Main.py -t True (--train True)')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('Main.py -t True (--train True)')
            sys.exit()
        elif opt in ("-t", "--train"):
            train = True if (arg.lower() == "true" ) else False
    
    if train:
        perform_face_classification()
    
    perform_face_detection()
    


if __name__ == "__main__":
    main(sys.argv[1:])