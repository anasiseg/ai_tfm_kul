import requests

URL = "http://127.0.0.1:5000/sharp/all"
        
# sending get request and saving the response as response object
r = requests.get(url = URL)

# extracting data in json format
ids = r.json()

for id in ids:
    URL = "http://127.0.0.1:5000/sharp/{id}/download".format(id=id)
            
    # sending get request and saving the response as response object
    r0 = requests.get(url = URL)

    # extracting data in json format
    harp_metadata = r0.json()
    print(harp_metadata)