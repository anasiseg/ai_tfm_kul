# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from src.database import DatabaseManager
from src.sunpyUtils import sunpyUtils
from src.imagesManager import ImageManager
from flask import Flask, request
from flask import Response
from io import BytesIO
import zipfile
from flask_restx import Api, Resource, fields
 
# Flask constructor takes the name of 
# current module (__name__) as argument.
app = Flask(__name__)
api = Api(app)

ddbbManager = DatabaseManager()
spUtils = sunpyUtils()
imgManager = ImageManager()

# The route() function of the Flask class is a decorator, 
# which tells the application which URL should call 
# the associated function.
ns = api.namespace("sharp", description="Sharp operations")
@ns.route('/<id>')
class Sharp(Resource):
    def get(self, id):#get_sharp_images
        result = ddbbManager.get_harp_image(id)
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for i, data in enumerate(result):
                    zip_file.writestr(str(i)+".jpeg",data.getvalue())
        return Response(zip_buffer.getvalue(),
                        mimetype='application/zip',
                        headers={'Content-Disposition': 'attachment;filename=your_filename.zip'})
    
    def post(self, id):
        # spUtils.reset()
        print(f'HARP NUMBER: {id}')
        data = request.json
        try:
            init_time = data['init_time']
            end_time = data['end_time']
            noaa = int(data['noaa'])
            ddbbManager.add_harp(int(id), init_time, end_time, noaa)
            return {'result': 'ok'}
        except:
            return {'result':'error'}

@ns.route('/all')
class SharpAll(Resource):
    def get(self):#get_all_sharp_images
        result = ddbbManager.get_all_harp_id()
        return result
    
@app.route('/sharp/<id>/download')
async def post(id):
    spUtils.reset()
    print(f'HARP NUMBER: {id}')
    try:
        init_time, end_time = ddbbManager.get_harp_by_id(id)
        result = spUtils.download_SHARP(id, 12, init_time.strftime('%Y-%m-%dT%H:%M:%S'), end_time.strftime('%Y-%m-%dT%H:%M:%S')) 
        df = spUtils.get_df()
        sequence = await spUtils.get_sequence(result['jsoc'])
        for index, row in df.iterrows():
            t = row['Date_Time']
            usflux= row['USFLUX']
            buf = imgManager.get_img_from_plot(sequence[row['id_result']])
            ddbbManager.add_harp_time_evolution(id, t, buf, usflux)
        return {'result': 'ok'}
    except Exception as error:
        return {'result':'error', 'msg': error}

ns_goes = api.namespace("goes", description="goes reports")
@ns_goes.route('/all')
class GOES(Resource):
    def get(self):#get_sharp_images
        tstart = "2018/10/28"
        tend = "2023/10/29"
        result = spUtils.get_goes(tstart, tend)
        return result
    
@ns_goes.route('/noaa/<id>')
class GOES_NOAA(Resource):
    def get(self, id):#get_all_sharp_images
        tstart = "2018/10/28"
        tend = "2023/10/29"
        try:
            result = {'result': 'ok', 'data': spUtils.get_SHARP_by_NOAA(id, tstart, tend)}
        except Exception as ex:
            print(ex)
            result = {'result': 'error', 'msg': str(ex)}
        return result

api.add_namespace(ns)
api.add_namespace(ns_goes)

# main driver function
if __name__ == '__main__':
    import sys
    import asyncio
    if sys.platform == "win32" and sys.version_info >= (3, 8, 0):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    # run() method of Flask class runs the application 
    # on the local development server.
    app.run()