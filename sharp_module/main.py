from src.sunpyUtils import sunpyUtils
from src.imagesManager import ImageManager
from src.database import DatabaseManager

spUtils = sunpyUtils()
imgManager = ImageManager()
ddbbManager = DatabaseManager()

result = ddbbManager.get_all_harp_id()
# array_HARP=[401, 2920, 377, 364, 384, 1953]
# ids = [e for e in result if e not in array_HARP]
for id in result:
    spUtils.reset()
    print(f'HARP NUMBER: {id}')
    try:
        init_time, end_time = ddbbManager.get_harp_by_id(id)
        result = spUtils.download_SHARP(id, 12, init_time.strftime('%Y-%m-%dT%H:%M:%S'), end_time.strftime('%Y-%m-%dT%H:%M:%S')) 
        df = spUtils.get_df()
        sequence = spUtils.get_sequence(result['jsoc'])
        for index, row in df.iterrows():
            t = row['Date_Time']
            usflux= row['USFLUX']
            buf = imgManager.get_img_from_plot(sequence[row['id_result']])
            ddbbManager.add_harp_time_evolution(id, t, buf, usflux)
    except Exception as error:
        print(error)
print("finished")