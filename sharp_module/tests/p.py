from sunpy.map import Map
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.visualization import ImageNormalize, SqrtStretch
import numpy as np

sequence = Map('data', sequence=True)
# sharp_map.peek(clip_interval=(1, 99.99)*u.percent)
# fig = plt.figure()
# ax = fig.add_subplot(projection=sequence.maps[0])
# ani = sequence.plot(axes=ax, norm=ImageNormalize(vmin=0, vmax=5e3, stretch=SqrtStretch()))
new_dimensions = [256, 128] * u.pixel
smap = sequence.maps[0].resample(new_dimensions)

figure = plt.figure(frameon=False)
ax = plt.axes([0, 0, 1, 1])
# Disable the axis
ax.set_axis_off()

# Plot the map.
# Since we are not interested in the exact map coordinates,
# we can simply use :meth:`~matplotlib.Axes.imshow`.
norm = smap.plot_settings['norm']
norm.vmin, norm.vmax = np.percentile(smap.data, [1, 99.9])
ax.imshow(smap.data,
          norm=norm,
          cmap=smap.plot_settings['cmap'],
          origin="lower")

plt.savefig('p.jpeg')

plt.show()  