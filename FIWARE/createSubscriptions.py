import requests
import json
import yaml

with open("configFIWARE.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    subscriptions = config["subscriptions"]
    header = config["header_json"]
    url = config["subscriptions_url"]

for subscription in subscriptions:
    subscription_json = json.dumps(subscription)
    print(subscription_json)
    response = requests.post(
        url, headers=header, data=subscription_json
    )  # To create entity we need to send post on appropriate URL adress
    # It also must have appropriate header and Json in specific format
    print(response.text)
    print(f"Created entity Subscription: {response.status_code}")
