import time
import json
import requests
import sys

def print_anywhere(message=""):
    print(message,file=sys.stderr)


def set_status(message="", x_next=[], fiware_url="127.0.0.1"):
    timestamp = str(time.time())
    # http://host.docker.internal:1026/ngsi-ld/v1/entities/urn:ngsi-ld:Process:PBO
    #final_url = "http://%s:1026/ngsi-ld/v1/entities/urn:ngsi-ld:Process:PBO/attrs"%(fiware_url)
    final_url = "http://host.docker.internal:1026/ngsi-ld/v1/entities/urn:ngsi-ld:Process:PBO/attrs"
    header = {"Content-Type": "application/ld+json"}

    update = {
        "Status": {
            "value": {"message": message, "x_next": x_next}
        },
        "@context": [
            "https://smartdatamodels.org/context.jsonld",
            "https://raw.githubusercontent.com/shop4cf/data-models/master/docs/shop4cfcontext.jsonld"
        ]

    }

    update_json = json.dumps(update)
    resp = "None"
    try:
        resp = requests.patch(final_url, headers=header, data=update_json)
    except requests.exceptions.ConnectionError:
        print_anywhere("Exception occurred")
        print_anywhere(resp)
    print_anywhere("Status updated.")

#
# if __name__ == '__main__':
#     set_status("ready for input", [2, 3, 4])
