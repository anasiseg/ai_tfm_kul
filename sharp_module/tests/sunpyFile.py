import os

import matplotlib.pyplot as plt

import astropy.units as u

from sunpy.map import Map
from sunpy.net import Fido
from sunpy.net import attrs as a
import pandas as pd
import datetime
import numpy as np
from astropy.time import Time
import io

# print(a.jsoc.Series)

email = "ana.simarrosegura@student.kuleuven.be"
# res = Fido.search(a.Time('2014-01-01T00:00:00', '2014-01-01T00:20:00'),
#                     a.jsoc.Series('hmi.sharp_720s '),
#                     a.jsoc.Notify(email),
#                     a.jsoc.PrimeKey("HARPNUM", 3520))2011.02.15_02:12:00_TAI
start_time = Time('2011-02-15T02:10:00', scale='utc', format='isot')
harp_num=377
result = Fido.search(a.Time(start_time - 1*u.h, start_time + 1*u.h),
                     a.Sample(2*u.hour),
                    #  a.Instrument("HMI"),
                     a.jsoc.Segment("Br"),
                     a.jsoc.Series("hmi.sharp_cea_720s"),
                     a.jsoc.PrimeKey("HARPNUM", harp_num),
                     a.jsoc.Notify(email))

# result = Fido.search(a.Time('2014-01-01T00:00:00', '2014-01-01T01:00:00'),a.jsoc.Notify(email),
#                   a.jsoc.Series('hmi.sharp_cea_720s')) 

harpData=False
harpImagePlot = True

if harpData:
    keydf={}
    def convert_to_datetime(x):
        return datetime.datetime.strptime(x,"%Y.%m.%d_%H:%M:%S_TAI")

    vfunc = np.vectorize(convert_to_datetime)
    keydf['T_REC'] = vfunc(np.array(result['jsoc']['T_REC']))

    for k in result['jsoc'].show('USFLUX','MEANGAM','MEANGBT','MEANGBZ','MEANGBH','MEANJZD','TOTUSJZ','MEANALP', 'MEANJZH', 'TOTUSJH', 'ABSNJZH', 'SAVNCPP', 'MEANPOT', 'TOTPOT', 'MEANSHR', 'SHRGT45').keys():
        keydf[k]=np.array(result['jsoc'][k])


    df= pd.DataFrame(keydf)
    df= df.sort_values(by="T_REC")

    print("Line graph: ") 

    num4Figures = int(len(df.keys())/4)
    numExtraFigure =len(df.keys())%4

    for i in range(num4Figures):
        fig, axs = plt.subplots(2, 2)
        key1=df.keys()[0+i*4]
        key2=df.keys()[1+i*4]
        key3=df.keys()[2+i*4]
        key4=df.keys()[3+i*4]
        axs[0, 0].plot(df["T_REC"], df[key1])
        axs[0, 0].set_title(key1)
        axs[0, 1].plot(df["T_REC"], df[key2])
        axs[0, 1].set_title(key2)
        axs[1, 0].plot(df["T_REC"], df[key3])
        axs[1, 0].set_title(key3)
        axs[1, 1].plot(df["T_REC"], df[key4])
        axs[1, 1].set_title(key4)
        # plt.plot(df["T_REC"], df["USFLUX"])

if harpImagePlot:
    file = Fido.fetch(result)
    # sharp_map = Map(file)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection=sharp_map)
    # sharp_map.plot(axes=ax, vmin=-1500, vmax=1500)

    # plt.show()  
    sequence = Map(file) #, sequence=True)
    hmimag = plt.get_cmap('hmimag')
    for fig in sequence:
        buf = io.BytesIO()
        # src=f'./data_images/HMI/HARP_{harp_num}_{str(fig._date_obs)}.jpeg'
        plt.imsave(buf, fig.data,cmap=hmimag,origin='lower',vmin=-3000,vmax=3000)

    # fig = plt.figure()
    # ax = fig.add_subplot(projection=sequence.maps[0])
    # ani = sequence.plot(axes=ax, norm=ImageNormalize(vmin=0, vmax=5e3, stretch=SqrtStretch()))

    # plt.show()
    # input1 = input()

print("")                