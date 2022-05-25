import shutil

import matlab.engine
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    filename = 'Copy_of_multipage_tif_resized.tif'
    eng = matlab.engine.start_matlab()
    [phi_0, masks, cell_ts, nhbd_ts, corrIm, smaller_ROIs, larger_ROI] = eng.demo(filename, nargout = 7)
    # convert to numpy
    phi_0 = np.asarray(phi_0)
    masks = np.asarray(masks)
    cell_ts = np.asarray(cell_ts)
    nhbd_ts = np.asarray(nhbd_ts)
    corrIm = np.asarray(corrIm)
    smaller_ROIs = np.asarray(smaller_ROIs)
    larger_ROI = np.asarray(larger_ROI)

    print(phi_0.shape)
    print(masks.shape)
    print(cell_ts.shape)
    print(nhbd_ts.shape)
    print(corrIm.shape)
    print(smaller_ROIs.shape)
    print(larger_ROI.shape)

    # save the footprints and temporal traces of each neurons as a png in a dedicated folder
    # make the dedicated folder and if already exists, delete the previous one
    dir = 'ABLE_results_' + str(filename)
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

    # set the intensity of the mask to 0 if it is false
    masks = np.ma.masked_where(masks == False, masks)

    for neuron_index in range(masks.shape[2]):
        fig, axes = plt.subplots(1, 2, figsize=(10, 2), gridspec_kw={'width_ratios': [1, 3]})

        # spatial footprint
        axes[0].imshow(corrIm, interpolation='nearest')
        axes[0].imshow(masks[:, :, neuron_index], cmap='jet', alpha=0.8)  # interpolation='none'
        axes[0].set_title("Spatial footprint", fontsize=5)
        axes[0].grid(False)
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        # temporal dynamics
        axes[1].set_title("Temporal trace", fontsize=5)
        axes[1].plot(cell_ts[neuron_index, :] - nhbd_ts[neuron_index, :] , label='neuron {0}'.format(neuron_index), color='blue')
        axes[1].set_ylabel("dF over F", fontsize=5)
        axes[1].set_xlabel("frame number", fontsize=5)
        axes[1].grid()

        name = dir + '/ABLE_results_' + str(neuron_index)
        plt.savefig(name, dpi=1200)
        # allows to not display the individual ones
        plt.close(fig)


    '''
    plt.imshow(corrIm, interpolation='nearest', cmap=plt.gray())
    for i in range(masks.shape[2]):
        plt.imshow(masks[:,:,i], cmap='jet', alpha=0.8)  # interpolation='none'
    plt.show()
    
    plt.plot(cell_ts[i,:])
    plt.plot(nhbd_ts[i,:])
    plt.plot(cell_ts[i,:] - nhbd_ts[i,:])
    plt.show()'''

