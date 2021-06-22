import speech_recognition as sr


class SpeechRecognition:
    def __init__(self):
        super().__init__()
        self.microphone = sr.Microphone()
        self.recognizer = sr.Recognizer()

    def listen_to_microphone(self):
        with self.microphone as src:
            self.recognizer.adjust_for_ambient_noise(src)
            return self.recognizer.listen(src)


    def get_text(self, voice):
        return self.recognizer.recognize_google(voice)



if __name__ == "__main__":
    
    speech_reg = SpeechRecognition()
    voice = speech_reg.listen_to_microphone()
    text = speech_reg.get_text(voice)
    print(text)