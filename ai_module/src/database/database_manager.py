from PIL import Image
from io import BytesIO

class DatabaseManager():

    def __init__(self):
        pass

    def get_all_harp_id(self):
        import requests
        # api-endpoint
        URL = "http://127.0.0.1:5000/sharp/all"
        
        # sending get request and saving the response as response object
        r = requests.get(url = URL)
        
        # extracting data in json format
        ids = r.json()
        return ids

    def get_harp_image(self, id_sharp=377):
        import requests
        import zipfile
        URL2='http://127.0.0.1:5000/sharp/{id}'.format(id=id_sharp)

        # sending get request and saving the response as response object
        r = requests.get(url = URL2)
        
        # extracting data in json format
        # data = r.json()
        z = zipfile.ZipFile(BytesIO(r.content))
        img_array=[]
        for i, img in enumerate(z.infolist()):
            img_buff= z.read(img)
            img_array.append(Image.open(BytesIO(img_buff)).convert('RGB'))
        return img_array

    
    # def transform_image(self, sharp_image_array):
    #     tensor = [Image.open(BytesIO(sharp_image)).convert('RGB') for sharp_image in sharp_image_array]
    #     return tensor
    
        

