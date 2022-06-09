import shutil
import matlab.engine
import numpy as np
import matplotlib.pyplot as plt
import os
from xml.etree.ElementTree import Element, tostring


class ABLE_class():

    def __init__(self, filename, radius, alpha, blur_radius, lambda_param, mergeCorr, metric, maxlt):

        # save the footprints and temporal traces of each neurons as a png in a dedicated folder
        # make the dedicated folder and if already exists, delete the previous one

        dir_general = 'RESULTS_' + str(filename)
        dir_general, temp = os.path.splitext(dir_general)  # removes the .tif
        if os.path.exists(dir_general):
            pass
        else:
            # the general directory does not exist, so create it
            os.makedirs(dir_general)

        dir = str(dir_general) + '/ABLE_results'
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

        # store the parameters in a dictionary:
        s = {'filename': filename,
             'radius': radius,
             'alpha': alpha,
             'blur_radius': blur_radius,
             'lambda_param': lambda_param,
             'mergeCorr': mergeCorr,
             'metric': metric,  # 0,
             'maxlt': maxlt
             }

        # convert the parameters to an xml file:
        # e stores the element instance
        e = self.dictionary_to_xml('parameters', s)
        completeName = os.path.join(dir, "parameters.xml")
        f = open(completeName, "wb")
        f.write(tostring(e))
        f.close()

        # filename = 'Copy_of_multipage_tif_resized.tif'
        eng = matlab.engine.start_matlab()
        [masks, cell_ts, nhbd_ts, corrIm, smaller_ROIs, larger_ROI] = eng.demo(filename, radius, alpha, blur_radius,
                                                                               lambda_param, mergeCorr, metric, maxlt,
                                                                               nargout=6)
        # convert to numpy
        self.masks = np.asarray(masks)
        self.cell_ts = np.asarray(cell_ts)
        self.nhbd_ts = np.asarray(nhbd_ts)
        self.corrIm = np.asarray(corrIm)
        smaller_ROIs = np.asarray(smaller_ROIs)
        larger_ROI = np.asarray(larger_ROI)

        '''print(masks.shape)
        print(cell_ts.shape)
        print(nhbd_ts.shape)
        print(corrIm.shape)
        print(smaller_ROIs.shape)
        print(larger_ROI.shape)'''

        # set the intensity of the mask to 0 if it is false
        self.masks = np.ma.masked_where(self.masks == False, self.masks)

        for neuron_index in range(self.masks.shape[2]):
            fig, axes = plt.subplots(1, 2, figsize=(10, 2), gridspec_kw={'width_ratios': [1, 3]})

            # spatial footprint
            axes[0].imshow(self.corrIm, interpolation='nearest')
            axes[0].imshow(self.masks[:, :, neuron_index], alpha=0.8)  # interpolation='none'
            axes[0].set_title("Spatial footprint", fontsize=5)
            axes[0].grid(False)
            axes[0].set_xticks([])
            axes[0].set_yticks([])

            # temporal dynamics
            axes[1].set_title("Temporal trace", fontsize=5)
            axes[1].plot(self.cell_ts[neuron_index, :] - self.nhbd_ts[neuron_index, :], label='neuron {0}'.format(neuron_index),
                         color='blue')
            axes[1].set_ylabel("dF over F", fontsize=5)
            axes[1].set_xlabel("frame number", fontsize=5)
            axes[1].grid()

            name = dir + '/ABLE_results_' + str(neuron_index)
            plt.savefig(name, dpi=1500)
            # allows to not display the individual plots when generating them
            plt.close(fig)

    # convert a simple dictionary
    # of key/value pairs into XML
    # credit: https://www.geeksforgeeks.org/turning-a-dictionary-into-xml-in-python/
    def dictionary_to_xml(self, tag, d):

        elem = Element(tag)
        for key, val in d.items():
            # create an Element
            # class object
            child = Element(key)
            child.text = str(val)
            elem.append(child)

        return elem

    def plot_summary(self):

        number_neurons_detected = self.masks.shape[2]

        for neuron_index in range(self.masks.shape[2]):
            ax1 = plt.subplot2grid((number_neurons_detected, 4), (neuron_index, 0), colspan=1)
            ax2 = plt.subplot2grid((number_neurons_detected, 4), (neuron_index, 1), colspan=3)

            # spatial footprint
            ax1.imshow(self.corrIm, interpolation='nearest')
            ax1.imshow(self.masks[:, :, neuron_index], alpha=0.8)  # interpolation='none'
            ax1.set_title("Spatial footprint", fontsize=5)
            ax1.grid(False)
            ax1.set_xticks([])
            ax1.set_yticks([])

            # temporal dynamics
            ax2.set_title("Temporal trace", fontsize=5)
            ax2.plot(self.cell_ts[neuron_index, :] - self.nhbd_ts[neuron_index, :], label='neuron {0}'.format(neuron_index),
                         color='blue')
            ax2.set_ylabel("dF over F", fontsize=5)
            ax2.set_xlabel("frame number", fontsize=5)
            ax2.grid()

        plt.show()



