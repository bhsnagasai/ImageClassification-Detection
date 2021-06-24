import face_recognition
import pickle
import cv2
import os

class FaceClassification:
    
    def __init__(self):
        super().__init__()
        self.dataset_path = "dataset"
        self.not_allowed_name = "premiere"
        self.save_path = os.path.join(self.dataset_path, "face_encodings")
        self.file_Paths = {}
        self.encodings = []
        self.encoding_names = []
        self.names = []
        
    
    def get_names(self):
        self.names = [ f.path.split("\\")[1] for f in os.scandir(self.dataset_path) if (f.is_dir() and  (self.not_allowed_name not in f.path.lower())) ]
        
    def get_image_paths(self):
        for name in self.names:
            for file in os.scandir(os.path.join(self.dataset_path, name)):
                if name in self.file_Paths:
                    self.file_Paths[name].append(file.path)
                else:
                    self.file_Paths[name] = [file.path]
                    
    def recognize_faces(self):
        for name in self.names:
            for file_path in self.file_Paths[name]:
                
                image = cv2.imread(file_path)
                rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                face_boxes = face_recognition.face_locations(rgb_img, model="hog")
                face_encodings = face_recognition.face_encodings(rgb_img, face_boxes)
                for encoding in face_encodings:
                    self.encoding_names.append(name)
                    self.encodings.append(encoding)

    
    def save_to_file(self):
        data = {"encodings": self.encodings, "names": self.encoding_names}
        file = open(self.save_path, "wb")
        file.write(pickle.dumps(data))
        file.close
        
        
if __name__ == "__main__":
    face_class = FaceClassification()
    face_class.get_names()
    face_class.get_image_paths()
    face_class.recognize_faces()
    face_class.save_to_file()
    