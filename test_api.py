import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "features": [100, 6, 443, 80, 1, 0, 0]
}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Raw Response:", response.text)