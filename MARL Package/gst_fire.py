import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import gstools as gs

model = gs.Exponential(
    latlon=True,
    temporal=True,
    var=1,
    len_scale=100,
    geo_scale=gs.KM_SCALE,
)

lat = lon = np.linspace(-80, 81, 50)
time = np.linspace(0, 777, 50)

pos, time = [lat, lon], [time]

srf = gs.SRF(model, seed=1234)
srf.structured(pos + time)
# srf.plot()

def _update_ani(time_step):
    im.set_array(srf.field[:, :, time_step].T)
    return (im,)


fig, ax = plt.subplots()
im = ax.imshow(
    srf.field[:, :, 0].T,
    cmap="Blues",
    interpolation="bicubic",
    origin="lower",
)
cbar = fig.colorbar(im)
cbar.ax.set_ylabel(r"Precipitation $P$ / mm")
ax.set_xlabel(r"$x$ / km")
ax.set_ylabel(r"$y$ / km")

ani = animation.FuncAnimation(
    fig, _update_ani, len(time), interval=100, blit=True
)

# from IPython.display import HTML

# HTML(ani.to_jshtml())
