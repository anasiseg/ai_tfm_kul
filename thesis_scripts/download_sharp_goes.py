import requests

URL = "http://127.0.0.1:5000/goes/all"
        
# sending get request and saving the response as response object
r = requests.get(url = URL)

# extracting data in json format
ids = r.json()

for id in ids:
    URL = "http://127.0.0.1:5000/goes/noaa/{id}".format(id=id)
            
    # sending get request and saving the response as response object
    r0 = requests.get(url = URL)

    # extracting data in json format
    result_noaa = r0.json()
    if result_noaa['result']=='error':
        print(result_noaa['msg'])
    else: 
        harp_metadata = result_noaa['data']
        try:
            # data to be sent to api
            API_ENDPOINT = "http://127.0.0.1:5000/sharp/{id}".format(id=harp_metadata[0]['harp_num'])
            data = {'init_time': harp_metadata[0]['init_time'],
                    'end_time': harp_metadata[0]['end_time'],
                    'noaa': harp_metadata[0]['noaa']}
            
            # sending post request and saving response as response object
            r1 = requests.post(url=API_ENDPOINT, json=data)
            print(r1.text)
        except:
            print("not post for {id} harp_metadata result:".format(id=id))
            print(harp_metadata)

print(ids)

"""
errors:
12158
12567
12173
12151
12056
12146
11934
12149
12130
12155
12172
12166
12051
12080
11900
12157
11989
12169
11904
"""