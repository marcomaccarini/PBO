"""
So we gets all entities from config and create them in FIWARE
"""
import requests
import json
import yaml


with open("configFIWARE.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    entities = config["entities"]
    header = config["header_json"]
    url = config["entities_url"]

for entity in entities:
    entity_json = json.dumps(entity)
    print(entity_json)
    response = requests.post(
        url, headers=header, data=entity_json
    )  # To create entity we need to send post on appropriate URL adress
    # It also must have appropriate header and Json in specific format
    print(f"Created entity {entity['id']}: {response.status_code}")
