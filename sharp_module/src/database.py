import psycopg2
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import base64

class DatabaseManager():

    def __init__(self):
        self.conn = psycopg2.connect(database="sunpyStorage",
                                host="192.168.16.1",
                                user="postgres",
                                password="asimarro",
                                port="5432")

        self.cursor = self.conn.cursor()

    def add_harp(self, number, start_time, end_time, noaa):
        try:
            table='harp_definition'
            self.cursor.execute("SELECT * FROM "+table+" WHERE harp_number="+str(number))
            select_result = self.cursor.fetchall()

            if len(select_result)==0:
                query = f"INSERT INTO {table} (harp_number, start_time, end_time, downloaded) VALUES ({number}, '{start_time}', '{end_time}', False)"
                self.cursor.execute(query)
                print('saving harp id {id}'.format(id=number))
        except Exception as ex:
            print('Errors on inserting harp_definition')
            print(ex)

        try:
            table='harp_phenomena'
            self.cursor.execute("SELECT * FROM "+table+" WHERE harp_number="+str(number)+" AND NOAA_id="+str(noaa))
            select_result = self.cursor.fetchall()
            if len(select_result)==0:
                self.cursor.execute("INSERT INTO "+table+" (NOAA_id, harp_number) VALUES (%s, %s)", (noaa, number))
                print('saving noaa id {id}'.format(id=noaa))
        except Exception as ex:
            print('Errors on inserting harp_phenomena')
            print(ex)

        try:
            self.persist_changes()
        except Exception as ex:
            print('Errors on saving changes')
            print(ex)

    def add_harp_time_evolution(self, number, time, image, usflux):
        table='harp_evolution'
        self.cursor.execute("INSERT INTO "+table+" (harp_number, evolution_time, evolution_image, usflux) VALUES (%s, %s, %s, %s)", (number, time, image.getvalue(), usflux))
        table='harp_definition'
        query = f"UPDATE {table} SET downloaded = TRUE WHERE harp_number = {number}"
        self.cursor.execute(query)
        self.persist_changes()

    
    def persist_changes(self):
        self.conn.commit()

    # def get_harp_image(self):
    #     table='harp_evolution'
    #     self.cursor.execute("SELECT * FROM "+table)
    #     select_result = self.cursor.fetchall()
    #     img = Image.open(BytesIO(select_result[0][3]))
    #     plt.imshow(img)

    def get_all_harp_id(self):
        table='harp_definition'
        self.cursor.execute("SELECT * FROM "+table+" WHERE harp_number IN (3730, 3740) ")#+" WHERE downloaded = False")
        select_result = self.cursor.fetchall()
        return [x[0] for x in select_result]
    
    def get_harp_by_id(self, id):
        table='harp_definition'
        self.cursor.execute("SELECT * FROM "+table+" WHERE harp_number ="+str(id))
        x = self.cursor.fetchone()
        return (x[1], x[2])

    def get_harp_image(self, id_sharp=377):
        table='harp_evolution'
        self.cursor.execute("SELECT * FROM "+table+" WHERE harp_number="+str(id_sharp))
        select_result = self.cursor.fetchall()
        sharp_image_array = [BytesIO(x[3]) for x in select_result]
        # img = [Image.open(BytesIO(sharp_image)) for sharp_image in sharp_image_array]
        return sharp_image_array