import pickle
import time
import cv2
import os
import face_recognition
from SpeechRecognition import SpeechRecognition


class FaceDetection:
    
    def __init__(self):
        super().__init__()
        self.dataset_path = "dataset"
        self.load_path = os.path.join(self.dataset_path, "face_encodings")
        self.haarcascade_path = "/data/haarcascade_frontalface_alt2.xml"
        self.casc_face_path = os.path.dirname(cv2.__file__) + self.haarcascade_path
        self.data = ""
        self.faceCascade = cv2.CascadeClassifier(self.casc_face_path)
        self.trackList = []
    
    def load_data(self):
        with open(self.load_path, 'rb') as pickle_file:
            self.data = pickle.load(pickle_file)
    
    
    def get_track_list(self):
        speech_reg = SpeechRecognition()
        voice = speech_reg.listen_to_microphone()
        speech_reg.get_text(voice)
        speech_reg.text_to_speech()
        data = speech_reg.extract_data()
        print(data)
        self.trackList = []
        if "follow" in data["Action"] or "track" in data["Action"]:
             self.trackList = data["PersonName"]
    
    def start_video_detection(self):    
        
        video = cv2.VideoCapture(0)
        
        while True:
            
            ret, frame = video.read()
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.faceCascade.detectMultiScale(gray_frame, scaleFactor = 1.1, minNeighbors=5, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)
        
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_frame)
            names = []
            
            for enc in encodings:
                matches = face_recognition.compare_faces(self.data["encodings"], enc)
                name = "unknown"
                if True in matches:
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    for i in matchedIdxs:
                        name = self.data["names"][i]
                        counts[name] = counts.get(name, 0) + 1
                    
                    name = max(counts, key=counts.get)
                    
                names.append(name)
                
                for ((x, y, w, h), name) in zip(faces, names):
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            
            cv2.imshow("Frame", frame)
            
            k = cv2.waitKey(1)
            
            if k & 0xFF == ord('q'):
                break
            if k & 0xFF == ord('v'):
                print("call to voice")
                self.get_track_list()

            
        
        video.release()
        cv2.destroyAllWindows()   
        
        
if __name__ == "__main__":
    face_dect = FaceDetection()
    face_dect.load_data()
    face_dect.start_video_detection()