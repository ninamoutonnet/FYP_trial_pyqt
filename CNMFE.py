import inscopix_cnmfe
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class CNMFE_class():

    def __init__(self, filename):

        # perform the CNMFE algporithm and store the results in the footprints and traces variables.
        # the CNMFE_class object requires the filename if the tiff stack on which it performs the CNMFE
        self.footprints, self.traces = inscopix_cnmfe.run_cnmfe(
            input_movie_path=filename,
            output_dir_path='output',
            output_filetype=0,
            average_cell_diameter=20,  # 7(default)->3, 15 -> 5, 20 -> 5
            min_pixel_correlation=0.8,
            min_peak_to_noise_ratio=10.0,
            gaussian_kernel_size=0,
            closing_kernel_size=0,
            background_downsampling_factor=2,
            ring_size_factor=1.4,
            merge_threshold=0.7,
            num_threads=4,
            processing_mode=2,
            patch_size=80,
            patch_overlap=20,
            output_units=1,
            deconvolve=0,
            verbose=1
        )

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
        #fig, axes = plt.subplots(number_neurons_detected, 2,  gridspec_kw={'width_ratios': [1, 3]})

        for neuron_index in range(self.footprints.shape[0]):
            # instantiate figure
            #  fig, axes = plt.subplots(1, 2, figsize=(10, 2), gridspec_kw={'width_ratios': [1, 3]})

            # spatial footprint
            '''axes[neuron_index, 0].imshow(self.footprints[neuron_index-1])
            axes[neuron_index, 0].set_title("Spatial footprint", fontsize=5)
            axes[neuron_index, 0].grid(False)
            axes[neuron_index, 0].set_xticks([])
            axes[neuron_index, 0].set_yticks([])

            # temporal dynamics
            axes[neuron_index, 1].set_title("Temporal trace", fontsize=5)
            axes[neuron_index, 1].plot(self.traces[neuron_index]-1, label='neuron {0}'.format(neuron_index), color='blue')
            axes[neuron_index, 1].set_ylabel("dF over noise", fontsize=5)
            axes[neuron_index, 1].set_xlabel("frame number", fontsize=5)
            axes[neuron_index, 1].grid()'''

            ax1 = plt.subplot2grid((number_neurons_detected, 4), (neuron_index, 0), colspan=1)
            ax2 = plt.subplot2grid((number_neurons_detected, 4), (neuron_index, 1), colspan=3)

            ax1.imshow(self.footprints[neuron_index-1])
            ax1.set_title("Spatial footprint", fontsize=5)

            ax2.plot(self.traces[neuron_index]-1, label='neuron {0}'.format(neuron_index), color='blue')
            ax2.set_title("Temporal trace", fontsize=5)
            ax2.set_ylabel("dF over noise", fontsize=5)
            ax2.set_xlabel("frame number", fontsize=5)
            ax2.grid()

        plt.tight_layout()
        plt.show()

    '''
    plot_footprints_on_grid(footprints)
    
    plot_composite_fov(footprints, colormap='gist_rainbow')
    
    plot_all_traces(traces, spacing=5, height_per_row=0.5)'''
