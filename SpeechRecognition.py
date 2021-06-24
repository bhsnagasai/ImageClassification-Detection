import speech_recognition as sr
from gtts import gTTS
import os
from playsound import playsound
from AzureLUIS import AzureLUIS


class SpeechRecognition:
    def __init__(self):
        super().__init__()
        self.microphone = sr.Microphone()
        self.recognizer = sr.Recognizer()
        self.azure_LUIS = AzureLUIS()
        self.text = "Welcome"
        self.lang = 'en'
        self.dirName = "dataset"
        self.fileName = "voice.mp3"
        
    def listen_to_microphone(self):
        with self.microphone as src:
            self.recognizer.adjust_for_ambient_noise(src)
            return self.recognizer.listen(src)


    def text_to_speech(self):
        gtts_obj = gTTS(text = self.text, slow = False)
        gtts_obj.save(os.path.join(self.dirName, self.fileName))
        playsound(os.path.join(self.dirName, self.fileName))
        print(self.text)
    
    def get_text(self, voice):
        self.text = self.recognizer.recognize_google(voice)

    def extract_data(self):
        self.azure_LUIS.get_prediction(self.text)
        return self.azure_LUIS.get_entities()

if __name__ == "__main__":
    
    speech_reg = SpeechRecognition()
    voice = speech_reg.listen_to_microphone()
    speech_reg.get_text(voice)
    speech_reg.text_to_speech()
    print(speech_reg.extract_data())