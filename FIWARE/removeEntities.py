"""
Delete all entities that are defined in config it sometimes may be usefull
"""
import requests
import yaml

with open("configFIWARE.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    entities = config["entities"]
    url = config["entities_url"]

for entity in entities:
    response = requests.delete(f'{url}/{entity["id"]}')
    print(f"Deleted entity {entity['id']}: {response.status_code}")
