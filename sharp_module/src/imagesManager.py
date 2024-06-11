from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import astropy.units as u
from sunpy.map import Map


class ImageManager():

    @staticmethod
    def resize_image(fig):
        new_dimensions = [256, 128] * u.pixel
        smap = fig.resample(new_dimensions)
        return smap
    
    @staticmethod
    def get_bytes_from_image(im):
    # file = "data_images\HMI\HARP_377_2011-02-15T01-10-12.300.jpeg"
    # im = Image.open(file)
        buf = BytesIO()
        im.save(buf, format='JPEG')
        byte_im = buf.getvalue()
        return byte_im
    
    @staticmethod
    def get_image_from_bytes(byte_im):
        img = Image.open(BytesIO(byte_im))
        # img.show()
        return img
    
    @staticmethod
    def get_img_from_plot(fig):
        hmimag = plt.get_cmap('hmimag')
        buf = BytesIO()
        fig = ImageManager.resize_image(fig)
        # src=f'./data_images/HMI/HARP_{harp_num}_{str(fig._date_obs)}.jpeg'
        plt.imsave(buf, fig.data, cmap=hmimag,origin='lower',vmin=-3000,vmax=3000)
        # plt.imshow(buf)
        return buf