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
          print(data['text'][0])

# Call the function to make the API call and handle the streaming response

call_api({"prompt": "The meaning of life is", "stream": True})

call_api({"prompt": "Has marvel made a good film", "stream": False})





