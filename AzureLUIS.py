import os
from dotenv import load_dotenv
from azure.cognitiveservices.language.luis.runtime import LUISRuntimeClient
from msrest.authentication import CognitiveServicesCredentials
from functools import reduce


class AzureLUIS:
    def __init__(self):
        super().__init__()
        load_dotenv()
        
        self.runtimeCredentials = CognitiveServicesCredentials(os.getenv('predictionKey'))
        self.clientRuntime = LUISRuntimeClient(endpoint=os.getenv('predictionEndpoint'), credentials=self.runtimeCredentials)
        self.predictionResponse = ""
        
    def get_prediction(self, query):
        self.predictionResponse = self.clientRuntime.prediction.get_slot_prediction(os.getenv('app_id'), "Production", { "query" : query })

    def get_entities(self):
        return self.predictionResponse.prediction.entities
        #print("Entities: {}".format (self.predictionResponse.prediction.entities))
    



if __name__ == "__main__":
    azure_LUIS = AzureLUIS()
    azure_LUIS.get_prediction(" follow ankit")
    azure_LUIS.get_entities()