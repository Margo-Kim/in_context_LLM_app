# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/03_inLoop.ipynb.

# %% auto 0
__all__ = ['inLoop']

# %% ../nbs/03_inLoop.ipynb 1
import pandas as pd
import requests 
import openai


# %% ../nbs/03_inLoop.ipynb 2
class inLoop:

    def __init__(self, data_file):
        self.data_csv = data_file

    
    def get_predictions(self, prompt):
        predictions = []
        for _, row in data_chunk.iterrows():
            text = row['Text']
            prediction = self.get_prediction_for_text(text)  # Implement this method
            predictions.append(prediction)
        return predictions
    
    def get_prediction_for_text(self, data):
        try:
         

            # API endpoint for ChatGPT
            url = "https://api.openai.com/v1/chat/completions"

            # Headers including the Authorization with your API key
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai_api_key}"
            }
             
            # Data payload for the request
            data_payload = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "You are a labeler and have to generate 5 labels. Each line has to have a label which is OTR(Opportunity to Respond), PRS(Praise),REP(Reprimand), NEU(None of the Above). And the result should only return the 5 labels and it should be seperated in comma. DO NOT INCLUDE ANY EXPLANATIONS. JUST LABELS SEPERATED IN COMMAS. ONLY GENERATE LABELS(OTR, PRS, REP, NEU) FOR EACH ROW. NO EXPLANATIONS. "},
                    {"role": "user", "content": f"Based on {data} labels: "}
                ]
            }

            # Make the POST request
            response = requests.post(url, headers=headers, json=data_payload)
            response_json = response.json()

            # Check if 'choices' is in the response and if it's not empty
            if 'choices' in response_json and response_json['choices']:
                # Accessing the first choice's message
                message = response_json['choices'][0]['message']['content']
                # Ensure the message is a string before calling strip()
                if isinstance(message, str):
                    return message.strip()
                else:
                    # Handle the case where message is not a string
                    return "Received non-string response: " + str(message)
            else:
                return "No choices in response or empty response: " + str(response_json)

            
        except Exception as e:
            return f"Error: {e}"

    
    def turn_dataframe(self, prediction, data_chunk):
        predicted_list = [item.strip() for item in prediction.split(',')]
        print(predicted_list)
        text_variable = data_chunk['Text'].to_list()
        print(text_variable)

        if len(predicted_list) < 5:
            predicted_list = predicted_list + ([None] * (5 - len(predicted_list)))
        if len(data_chunk) < 5 :
            text_variable = text_variable + ([None] * (5 - len(data_chunk)))
        df = pd.DataFrame({'Text': text_variable, 'Label' : predicted_list})
        print(df)
        return df
    
    
    def update_labels(self, start_index, revised_labels):
        for i, label in enumerate(revised_labels):
            self.data_csv.at[start_index + i, 'Label'] = label

    def get_updated_data(self):
        return self.data_csv

    def get_predictions(self, prompt):
        try:
            

            # API endpoint for ChatGPT
            url = "https://api.openai.com/v1/chat/completions"

            # Headers including the Authorization with your API key
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai_api_key}"
            }

            # Data payload for the request
            data_payload = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "prompt"},
                    {"role": "user", "content": f"Based on {self.dataset_string} solve the Question: {prompt}"}
                ]
            }

            # Make the POST request
            response = requests.post(url, headers=headers, json=data_payload)
            response_json = response.json()

            # Check if 'choices' is in the response and if it's not empty
            if 'choices' in response_json and response_json['choices']:
                # Accessing the first choice's message
                message = response_json['choices'][0]['message']
                # Ensure the message is a string before calling strip()
                if isinstance(message, str):
                    return message.strip()
                else:
                    # Handle the case where message is not a string
                    return "Received non-string response: " + str(message)
            else:
                return "No choices in response or empty response: " + str(response_json)

        except Exception as e:
            return f"Error: {e}"

