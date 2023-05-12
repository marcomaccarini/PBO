import requests
import json

# Nie mam pojęcia czemu ale czasami trzeba odpalić kilka razy, żeby usunąc wszystkie entities
type_ = "Task"

response = requests.get(f"http://localhost:1026/ngsi-ld/v1/subscriptions")

print(response.status_code)
subscriptions = json.loads(response.text)


for s in subscriptions:
    id_ = s["id"]
    print(s)
    # It is to delete only subscrpitions that we have created previously. To be 100% sure that we delete our subscription I filter
    # them using desciprion attribute
    if s["description"] in ["Notify me when an inspection is made", "Notify me when a new photo is made","Notify me when a task is made"]:
        print(id)
        response = requests.delete(
            f"http://localhost:1026/ngsi-ld/v1/subscriptions/{id_}")
        print(response.status_code)
