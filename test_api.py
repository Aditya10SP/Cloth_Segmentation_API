import requests

url = "http://127.0.0.1:5001/segment-cloth"
files = {"file": open("sample.png", "rb")}

response = requests.post(url, files=files)

# Save the segmented image locally
if response.status_code == 200:
    with open("segmented_output.png", "wb") as f:
        f.write(response.content)
    print("Segmented image saved as segmented_output.png")
else:
    print("Error:", response.json())
