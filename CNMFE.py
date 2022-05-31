import os
import shutil
import inscopix_cnmfe
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from xml.etree.ElementTree import Element,tostring


class CNMFE_class():

    def __init__(self, filename, average_cell_diameter_IN, min_pixel_correlation_IN, min_peak_to_noise_ratio_IN,
                 gaussian_kernel_size_IN, closing_kernel_size_IN, background_downsampling_factor_IN,
                 ring_size_factor_IN, merge_threshold_IN, num_threads_IN, processing_mode_IN,
                 patch_size_IN, patch_overlap_IN,  deconvolve_IN, output_units_IN, output_filetype_IN, verbose_IN):

        # store the parameters in a dictionary:
        s = {'input_movie_path': filename,
             'output_dir_path': 'output',
             'output_filetype': output_filetype_IN,
             'average_cell_diameter': average_cell_diameter_IN,  # 7(default)->3, 15 -> 5, 20 -> 5
             'min_pixel_correlation': min_pixel_correlation_IN,  # 0.8
             'min_peak_to_noise_ratio': min_peak_to_noise_ratio_IN,  # 10.0,
             'gaussian_kernel_size': gaussian_kernel_size_IN,  # 0,
             'closing_kernel_size': closing_kernel_size_IN,  # ,0
             'background_downsampling_factor': background_downsampling_factor_IN,  # 2,
             'ring_size_factor': ring_size_factor_IN,  # 1.4,
             'merge_threshold': merge_threshold_IN,  # 0.7,
             'num_threads': num_threads_IN,  # 4,
             'processing_mode': processing_mode_IN,  # 2,
             'patch_size': patch_size_IN,  # 80,
             'patch_overlap': patch_overlap_IN,  # 20,
             'output_units': output_units_IN,  # 1,
             'deconvolve': deconvolve_IN,  # 0,
             'verbose': verbose_IN,  # 1
             }

        # convert the parameters to an xml file:
        # e stores the element instance
        e = self.dictionary_to_xml('parameters', s)
        f = open("parameters.xml", "wb")
        f.write(tostring(e))
        f.close()

        # perform the CNMFE algporithm and store the results in the footprints and traces variables.
        # the CNMFE_class object requires the filename if the tiff stack on which it performs the CNMFE
        self.footprints, self.traces = inscopix_cnmfe.run_cnmfe(
            input_movie_path=filename,
            output_dir_path='output',
            output_filetype= output_filetype_IN,
            average_cell_diameter= average_cell_diameter_IN,  # 7(default)->3, 15 -> 5, 20 -> 5
            min_pixel_correlation= min_pixel_correlation_IN, # 0.8
            min_peak_to_noise_ratio=min_peak_to_noise_ratio_IN, #10.0,
            gaussian_kernel_size=gaussian_kernel_size_IN, #0,
            closing_kernel_size=closing_kernel_size_IN, #,0
            background_downsampling_factor= background_downsampling_factor_IN, # 2,
            ring_size_factor=ring_size_factor_IN, #1.4,
            merge_threshold= merge_threshold_IN, # 0.7,
            num_threads=num_threads_IN, #4,
            processing_mode=processing_mode_IN, #2,
            patch_size=patch_size_IN, #80,
            patch_overlap=patch_overlap_IN, #20,
            output_units=output_units_IN, #1,
            deconvolve=deconvolve_IN, #0,
            verbose=verbose_IN, #1
        )


        # save the footprints and temporal traces of each neurons as a png in a dedicated folder
        # make the dedicated folder and if already exists, delete the previous one
        dir = 'CNMFE_results_' + str(filename)
        dir, temp = os.path.splitext(dir) # removes the .tif
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

        for neuron_index in range(self.footprints.shape[0]):
            fig, axes = plt.subplots(1, 2, figsize=(10, 2), gridspec_kw={'width_ratios': [1, 3]})

            # spatial footprint
            axes[0].imshow(self.footprints[neuron_index - 1])
            axes[0].set_title("Spatial footprint", fontsize=5)
            axes[0].grid(False)
            axes[0].set_xticks([])
            axes[0].set_yticks([])

            # temporal dynamics
            axes[1].set_title("Temporal trace", fontsize=5)
            axes[1].plot(self.traces[neuron_index] - 1, label='neuron {0}'.format(neuron_index), color='blue')
            axes[1].set_ylabel("dF over noise", fontsize=5)
            axes[1].set_xlabel("frame number", fontsize=5)
            axes[1].grid()

            name = dir + '/CNMFE_results_' + str(neuron_index)
            plt.savefig(name, dpi=1200)
            # allows to not display the individual ones
            plt.close(fig)


    def plot_footprints_on_grid(self, footprints, n_cols=10):
        '''
        Plots all footprints on a grid of axes.
        User specifies number of desired columns

        Args:
            footprints (array):
                n_cells x rows x columns array of footprints
            n_cols (int, optional):
                number of columns in plot grid (defaults to 10).
        Returns:
            tuple of fig, ax
                fig = matplotlib figure handle
                ax = array of matplotlib axes handles
        '''

        n_rows = int(np.ceil(np.shape(footprints)[0] / n_cols))

        fig, ax = plt.subplots(
            n_rows,
            n_cols,
            figsize=(12, 1.6 * n_rows),
            sharex=True,
            sharey=True
        )
        axes = ax.ravel()

        for i in range(len(ax.flatten())):
            try:
                axes[i].imshow(footprints[i], cmap='gray')
                axes[i].set_title("cell {}".format(str(i).zfill(2)))
                axes[i].grid(False)
                axes[i].set_xticks([])
                axes[i].set_yticks([])
            except IndexError:
                # turn off axes for where cell doesn't exist
                axes[i].axis('off')

        fig.tight_layout()
        fig.subplots_adjust(wspace=0.025, hspace=0.2, top=0.93)
        fig.patch.set_facecolor('white')
        fig.suptitle('all spatial footprints', fontweight='bold');

        plt.show()

    def plot_composite_fov(self, footprints, colormap='gist_rainbow'):

        '''
        Plots all footprints on a single composite field of view.

        Args:
            footprints (array):
                n_cells x rows x columns array of footprints
            colormap (str, optional)
                Colormap to use. Each cell will be assigned a random color from this map (defaults to 'gist_rainbow').
        Returns:
            tuple of fig, ax
                fig = matplotlib figure handle
                ax = matplotlib axis handle
        '''

        fig, ax = plt.subplots(figsize=(15, 15))

        # start with an array of zeros
        composite_fov = np.zeros((footprints.shape[1], footprints.shape[2], 3))
        cmap_vals = cm.get_cmap(colormap)

        np.random.seed(0)
        for cell_id in range(footprints.shape[0]):
            # select a random color for this cell
            color = cmap_vals(np.random.rand())

            # assign the color to each of the three channels, normalized by the footprint peak
            for color_channel in range(3):
                composite_fov[:, :, color_channel] += color[color_channel] * footprints[cell_id] / np.max(
                    footprints[cell_id])

        # set all values > 1 (where cells overlap) to 1:
        composite_fov[np.where(composite_fov > 1)] = 1

        # show the image
        ax.imshow(composite_fov)

        # annotate each cell with a label centered at its peak
        for cell_id in range(footprints.shape[0]):
            peak_loc = np.where(footprints[cell_id] == np.max(footprints[cell_id]))
            ax.text(
                peak_loc[1][0],
                peak_loc[0][0],
                'cell {}'.format(str(cell_id).zfill(2)),
                color='white',
                ha='center',
                va='center',
                fontweight='bold',
            )

        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
        fig.patch.set_facecolor('white')
        fig.suptitle('composite field of view', fontweight='bold');
        plt.show()
        return fig, ax

    def plot_all_traces(self,traces, spacing=5, height_per_row=0.5, colormap='gist_rainbow'):
        '''
        Plots all traces on a single axis

        Args:
            traces (Pandas.DataFrame):
                Dataframe of all traces with columns Frame and one column for each cell
            spacing (int, optional):
                Vertical spacing between traces (defaults to 5).
            height_per_row (float, optional)
                Vertical height in inches devoted to each cell in the plot (defaults to 0.5).
            colormap (str, optional)
                Colormap to use. Each row will be a random color from this map (defaults to 'gist_rainbow').
        Returns:
            tuple of fig, ax
                fig = matplotlib figure handle
                ax = matplotlib axis handle
        '''

        fig, ax = plt.subplots(figsize=(15, height_per_row * self.footprints.shape[0]))

        num_cells = traces.shape[0]
        cell_names = ['cell ' + str(i).zfill(2) for i in range(num_cells)]
        cmap_vals = cm.get_cmap(colormap)
        np.random.seed(0)
        for cell_index in range(num_cells):
            ax.plot(
                range(traces.shape[1]),
                traces[cell_index] + -1 * cell_index * spacing - traces[cell_index][0],
                linewidth=3,
                alpha=0.75,
                color=cmap_vals(np.random.rand())
            )

        ax.set_ylim(-1 * spacing * len(cell_names) - spacing, 0 + spacing * 4)
        ax.set_yticks(np.arange(-1 * spacing * (len(cell_names) - 1), 0 + spacing, spacing));
        ax.set_yticklabels(cell_names[::-1]);
        ax.set_xlim(0, traces.shape[1] - 1)
        ax.set_xlabel('time (s)')

        for side in ['left', 'top', 'right']:
            ax.spines[side].set_color('white')

        fig.patch.set_facecolor('white')
        fig.tight_layout()
        fig.subplots_adjust(top=0.98)
        fig.suptitle('all trace timeseries', fontweight='bold');
        plt.show()
        return fig, ax

    def plot_summary(self):
        number_neurons_detected = self.footprints.shape[0]
        print('NEURON NB: ', number_neurons_detected)

        for neuron_index in range(self.footprints.shape[0]):

            ax1 = plt.subplot2grid((number_neurons_detected, 4), (neuron_index, 0), colspan=1)
            ax2 = plt.subplot2grid((number_neurons_detected, 4), (neuron_index, 1), colspan=3)

            ax1.imshow(self.footprints[neuron_index-1])
            ax1.set_title("Spatial footprint", fontsize=12)

            ax2.plot(self.traces[neuron_index]-1, label='neuron {0}'.format(neuron_index), color='blue')
            ax2.set_title("Temporal trace", fontsize=12)
            ax2.set_ylabel("dF over noise", fontsize=12)
            ax2.set_xlabel("frame number", fontsize=12)
            ax2.grid()

        plt.show()

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
