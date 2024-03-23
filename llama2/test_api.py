import requests
import json

# url = "https://as6xb.apps.beam.cloud/generate"
# payload = {}
# headers = {
#   "Accept": "*/*",
#   "Accept-Encoding": "gzip, deflate",
#   "Authorization": "Basic NzAyMmQyOTRkMjhiZjc2MmQzOGE0YzU2ODNkY2UzMjI6OThhM2E5NDFmMGNmZGU2NWM5MjMyOWExYWZkZjZmNjI=",
#   "Connection": "keep-alive",
#   "Content-Type": "application/json"
# }

def call_api(payload):
    # Define the API endpoint
    api_endpoint = "https://as6xb.apps.beam.cloud/generate"

    # Define the authorization token
    headers = {
        "Authorization": "Basic NzAyMmQyOTRkMjhiZjc2MmQzOGE0YzU2ODNkY2UzMjI6OThhM2E5NDFmMGNmZGU2NWM5MjMyOWExYWZkZjZmNjI=",
        "Content-Type": "application/json"
        # Add any other headers if required
    }

    stream = payload.get("stream", False)

    if stream:
      # Make a POST request to the API endpoint
      response = requests.post(api_endpoint, json=payload, headers=headers, stream=True)

      # Handle the streaming response
      if response.status_code == 200:
          # Iterate over the streaming response content
          for chunk in response.iter_content(chunk_size=None):
              # Process the chunk (assuming it's JSON)
              chunk = chunk.decode("utf-8").rstrip("\0")
              data = json.loads(chunk)
              print(data['text'][0])  # Do whatever processing needed with the data
      else:
          print("Error:", response.status_code, response.reason)
    else:
      # Make a POST request to the API endpoint
      response = requests.post(api_endpoint, json=payload, headers=headers)

      # Handle the non-streaming response
      if response.status_code == 200:
          data = response.json()
          return data['text'][0].strip()  # Return the response text

# Call the function to make the API call and handle the streaming response

# call_api({"prompt": "[INST]Write a narrative section on the importance of being earnest [/INST]", "stream": False, "sampling_params": {"max_tokens": 4096, "temperature": 0.8, "repetition_penalty": 1.15, "top_p": 1, "min_p": 0.1}, "lora": "science_fiction"})

prompt = "[INST]You are a renowned writer specialising in the genre of Science Fiction. You are able to create engaging narratives following a three act structure. Using the [Story Structure] outline, fill the [Story Events] suitable for the story as outlined in the premise.\n[Story Structure]\nSetup\n1.1 Exposition. The status quo or ‘ordinary world’ is established.\n1.2 Inciting Incident. An event that sets the story in motion.\n1.3 Plot Point A. The protagonist decides to tackle the challenge head-on. They ‘cross the threshold,’ and the story is now truly moving.\n\nConfrontation\n2.1 Rising Action. The story's true stakes become clear; our hero grows familiar with their ‘new world’ and has their first encounters with some enemies and allies.\n2.2 Midpoint. An event that upends the protagonist’s mission.\n2.3 Plot Point B. In the wake of the disorienting midpoint, the protagonist is tested — and fails. Their ability to succeed is now in doubt.\n\nResolution\n3.1 Pre Climax. The night is darkest before dawn. The protagonist must pull themselves together and choose between decisive action and failure.\n3.2 Climax. They faces off against her antagonist one last time. Will they prevail?\n3.3 Denouement. All loose ends are tied up. The reader discovers the consequences of the climax. A new status quo is established.\n[/Story Structure]\n\nPremise: A intrepid young interplanetary miner discovers that he is a clone and that the Martian government has bought the rights to his body.\n[Story Events]\nSetup\n\n1.1 Steve wakes up in the work dormitory. He has a strange feeling he can't shake. It's as if he wasn't alive an hour ago.\n1.2 \n1.3 \n\nConfrontation\n\n2.1 \n2.2 \n2.3 \n\nResolution\n\n3.1 \n3.2 \n3.3 \n[/Story Events]\n\nCreate a single event for the plot point 1.2, keep it concise and avoid repeating previous plot points.[/INST] 1.2 Inciting Incident:"
sampling_params = {"max_tokens": 4096, "temperature": 0.6, "stop": ["\n"], "repetition_penalty": 1.15, "top_p": 1, "min_p": 0.1}

call_api({"prompt": prompt, "stream": False, "sampling_params": sampling_params})



