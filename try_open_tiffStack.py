import sys
import matlab.engine
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print('hi')
    eng = matlab.engine.start_matlab()
    [phi_0, masks, cell_ts, nhbd_ts] = eng.demo('average_tif_resized.tif', nargout = 4)
    # convert to numpy
    phi_0 = np.asarray(phi_0)
    masks = np.asarray(masks)
    cell_ts = np.asarray(cell_ts)
    nhbd_ts = np.asarray(nhbd_ts)

    print(phi_0.shape)
    print(masks.shape)
    print(cell_ts.shape)
    plt.plot(cell_ts[1,:])
    plt.show()
    print(nhbd_ts.shape)




