import os
import io
import matplotlib.pyplot as plt

import astropy.units as u
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.map import Map
from astropy.time import Time

import pandas as pd
import datetime
import numpy as np


class sunpyUtils():
    def __init__(self):
        print("class created")
        self.email = "ana.simarrosegura@student.kuleuven.be"
        self.df = {} 

    def get_SHARP_Info(self, HARP_NUM):
        result = Fido.search(a.jsoc.Series("hmi.sharp_cea_720s"),
                            a.jsoc.PrimeKey("HARPNUM", HARP_NUM),
                            a.jsoc.Notify(self.email),
                            a.jsoc.Segment("Br"))
        vfunc = np.vectorize(self.convert_to_datetime)
        times = vfunc([result['jsoc']['T_REC'][0], result['jsoc']['T_REC'][-1]])
        return (times[0], times[-1])


    def download_SHARP(self, HARP_NUM, time_hout_step, init_time, end_time):
        start_time = Time(init_time, scale='utc', format='isot')
        end_time = Time(end_time, scale='utc', format='isot')
        result = Fido.search(a.Time(start_time, end_time),
                            a.Sample(time_hout_step*u.hour),
                            a.jsoc.Series("hmi.sharp_cea_720s"),
                            a.jsoc.PrimeKey("HARPNUM", HARP_NUM),
                            a.jsoc.Notify(self.email),
                            a.jsoc.Segment("Br"))
        
        keydf={}
        vfunc = np.vectorize(self.convert_to_datetime)
        keydf['T_REC'] = vfunc(np.array(result['jsoc']['T_REC']))

        for k in result['jsoc'].show('USFLUX','MEANGAM','MEANJZH','SHRGT45','MEANSHR').keys():
            keydf[k]=np.array(result['jsoc'][k])


        self.df= pd.DataFrame(keydf)
        self.df = self.df.dropna()
        self.df['id_result'] = self.df.index
        self.df = self.df.reset_index(drop=True)
        self.df= self.df.sort_values(by="T_REC")
        self.df['Date_Time']= self.df['T_REC'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        return result
    
    def get_df(self):
        return self.df

    def convert_to_datetime(self, x):
        return datetime.datetime.strptime(x,"%Y.%m.%d_%H:%M:%S_TAI")
    
    def plot_SHARP_graphs(self):
        num4Figures = int(len(self.df.keys())/4)
        numExtraFigure =len(self.df.keys())%4
        for i in range(num4Figures):
            fig, axs = plt.subplots(2, 2)
            key1=self.df.keys()[1+i*4]
            key2=self.df.keys()[2+i*4]
            key3=self.df.keys()[3+i*4]
            key4=self.df.keys()[4+i*4]
            axs[0, 0].plot(self.df["T_REC"], self.df[key1])
            axs[0, 0].set_title(key1)
            axs[0, 1].plot(self.df["T_REC"], self.df[key2])
            axs[0, 1].set_title(key2)
            axs[1, 0].plot(self.df["T_REC"], self.df[key3])
            axs[1, 0].set_title(key3)
            axs[1, 1].plot(self.df["T_REC"], self.df[key4])
            axs[1, 1].set_title(key4)
            fig.show()

    def get_sequence(self, result):
        file = Fido.fetch(result) 
        sequence = Map(file) #, sequence=True)
        return sequence
    
    def reset(self):
        del self.df


    def get_goes(self,tstart, tend):
        event_type = "FL"
        result = Fido.search(a.Time(tstart, tend),
                            a.hek.EventType(event_type),
                            a.hek.FL.GOESCls > "M1.0",
                            a.hek.OBS.Observatory == "GOES")
        hek_results = result["hek"]
        filtered_results = hek_results["event_starttime", "event_peaktime",
                               "event_endtime", "fl_goescls", "ar_noaanum"]
        by_magnitude = sorted(filtered_results, key=lambda x: ord(x['fl_goescls'][0]) + float(x['fl_goescls'][1:]), reverse=True)

        result = []
        for flare in by_magnitude:
            result.append(int(flare['ar_noaanum']))
        
        result_list = list(dict.fromkeys(result))  
        if 0 in result_list: result_list.remove(0)
        
        return result_list
    
    def get_SHARP_by_NOAA(self, NOAA, tstart, tend):
        result = Fido.search(a.jsoc.Series("hmi.sharp_cea_720s"),
                             a.Time(tstart, tend),
                            a.jsoc.Keyword('NOAA_AR') == NOAA,
                            a.jsoc.Notify(self.email),
                            a.jsoc.Segment("Br"))
        if len(result['jsoc'])==0:
            raise Exception(f"Not SHARPS are linked to NOAA number {NOAA}") 
        r = list(result['jsoc']['HARPNUM'])
        r=list(dict.fromkeys(r))
        vfunc = np.vectorize(self.convert_to_datetime)
        final_result=[]
        for harp in r:
            result_HARP = result['jsoc'][result['jsoc']['HARPNUM']==harp]
            times = vfunc([result_HARP['T_REC'][0], result_HARP['T_REC'][-1]])
            final_result.append({'harp_num': int(harp), 'noaa':NOAA, 'init_time': times[0].strftime('%Y-%m-%d %H:%M:%S'), 'end_time':times[-1].strftime('%Y-%m-%d %H:%M:%S')})

        return final_result
    
        