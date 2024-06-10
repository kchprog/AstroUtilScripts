import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse

## use mplcursors library to make the graph more interactive
import mplcursors

## Imports the Lomb-Scargle algorithm from the astropy library
## TODO: add functionality

from matplotlib.ticker import MultipleLocator, FuncFormatter, ScalarFormatter
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
from matplotlib.ticker import MultipleLocator

"""
    Code written by Kevin Chen for the purpose of AST251 planetary transfer research project. Some debugging assistance
    for the code included has been provided by the ChatGPT 4 LLM 
"""

def read_csv(file_path: str):
    """Read CSV file and return time and brightness as NumPy arrays."""
    time = []
    brightness_fractional = []
    brightness_uncertainty = []
    star_metadata = None

    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        ## Get information from the first line:
        star_metadata = next(reader)
        ## skip the separator
        next(reader)
        ## get information from the third line (column labels); not sure if this is needed, but retained.
        data_labels = next(reader)
        print(data_labels)
        ## Parse CSV file: for each 
        for row in reader:
            time.append(float(row[0]))
            brightness_fractional.append(float(row[3]))
            brightness_uncertainty.append(float(row[4]))
    
    return np.array(time), np.array(brightness_fractional), np.array(brightness_uncertainty), star_metadata


def find_spikes(time, brightness, prominence_min=0.00005):
    """Find significant spikes (minima) in the brightness data."""
    # Invert brightness to find minima as peaks
    inverted_brightness = -brightness
    
    # Find peaks in the inverted brightness
    peaks, properties = find_peaks(inverted_brightness, prominence=prominence_min, )
    
    # Filter peaks by prominence
    prominences = properties['prominences']
    significant_peaks = peaks[prominences >= prominence_min]
    
    # Find the deepest data point in each dip group
    dips = []
    for peak in significant_peaks:
        dips.append(peak)
    
    print("Identified dips:", dips)
    return np.array(dips)

def identify_gaps(time, threshold=50):
    """
        Identify periods where there are gaps between readings larger than the threshold.
        VARS: threshold is the number of days which we consider as a hard cutoff;
        anything shorter won't be labeled as a gap.
    """
    gaps = []
    for i in range(1, len(time)):
        if time[i] - time[i-1] > threshold:
            gaps.append((time[i-1], time[i]))
    return gaps


def plot_light_curve(time, brightness, brightness_uncertainty, metadata, peaks, file_path):
    """Plot the light curve using Matplotlib and label periodic variations and spikes."""
    plt.figure(figsize=(10, 6))

    plt.errorbar(time, brightness, yerr=brightness_uncertainty, fmt='o', color='black', markersize=1, label='Brightness (fractional)', capsize=2, elinewidth=0.25, alpha=0.2)
    
    plt.plot(time, brightness, marker='o', linestyle='None', color='b', markersize=1, label='Brightness (fractional)')

    ## Highlight significant periods and regions without readings with gray
    gaps = identify_gaps(time)

    for start, end in gaps:
        plt.axvspan(start, end, color='red', alpha=0.1)

    ## Adds a scatter plot over the actual graph for the lowest of the dips
    sc_dips = plt.scatter(time[peaks], brightness[peaks], color='red', zorder=5)

    ## Add interactive annotations with mplcursors
    cursor = mplcursors.cursor(sc_dips, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        idx = sel.index
        date = time[peaks][idx]
        intensity = brightness[peaks][idx]
        sel.annotation.set(text=f'Frac. Brightness: {intensity:.6f}\nT={date:.2f} days')
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)
        
    plt.xlabel('Time (Days)')
    plt.ylabel('Brightness (Fractional)')
    plt.title('Light Curve from File ' + file_path)
    plt.suptitle(metadata, fontsize=10)  # Display metadata as subtitle
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_transit_curve(time, brightness, brightness_uncertainty, metadata, date_start, date_end, name="PLACEHOLDER"):
    """Plot the light curve using Matplotlib and label periodic variations and spikes."""
    
    # Mask to filter the data between date_start and date_end
    mask = (time >= date_start) & (time <= date_end)
    zoomed_time = time[mask]
    zoomed_brightness = brightness[mask]
    zoomed_uncertainty = brightness_uncertainty[mask]

    # Determine start and end ticks for x-axis
    start_tick = np.floor(date_start * 2) / 2
    end_tick = np.ceil(date_end * 2) / 2
    plt.xticks(np.arange(start_tick, end_tick + 0.5, 0.5))

    # Set to avoid using scientific notation on axis labels
    plt.gca().xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    plt.gca().xaxis.get_major_formatter().set_scientific(False)
    plt.gca().xaxis.set_minor_locator(MultipleLocator(0.1))
    plt.minorticks_on()

    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    plt.gca().yaxis.get_major_formatter().set_scientific(False)
    plt.gca().yaxis.set_minor_locator(MultipleLocator(0.00001))

    # Plot the light curve with error bars
    scatter = plt.errorbar(zoomed_time, zoomed_brightness, yerr=zoomed_uncertainty, fmt='o', color='blue', markersize=2, 
                           label='Brightness (fractional)', capsize=2, elinewidth=1, alpha=1, ecolor='lightgray')

    # Detect the transit curve
    avg_brightness = np.mean(zoomed_brightness)
    std_brightness = np.std(zoomed_brightness)
    transit_threshold = avg_brightness - 0.0001 * std_brightness  # Threshold for detecting transit (customize as needed)

    transit_mask = zoomed_brightness < transit_threshold
    transit_times = zoomed_time[transit_mask]
    transit_brightness = zoomed_brightness[transit_mask]

    # Calculate average brightness on either side of the transit
    if len(transit_times) > 0:
        left_mask = zoomed_time < transit_times[0]
        right_mask = zoomed_time > transit_times[-1]

        left_avg_brightness = np.mean(zoomed_brightness[left_mask])
        right_avg_brightness = np.mean(zoomed_brightness[right_mask])
    else:
        left_avg_brightness = np.nan
        right_avg_brightness = np.nan

    # Calculate the lowest value in the transit
    if len(transit_brightness) > 0:
        lowest_brightness = np.min(transit_brightness)
    else:
        lowest_brightness = np.nan

    # Display the calculated values on the plot
    textstr = '\n'.join((
        f'Avg Brightness Left: {left_avg_brightness:.6f}',
        f'Avg Brightness Right: {right_avg_brightness:.6f}',
        f'Lowest Brightness: {lowest_brightness:.6f}',
    ))

    plt.gcf().text(0.05, 0.90, textstr, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    # Set plot labels and title
    plt.xlabel('Time (Days)')
    plt.ylabel('Brightness (Fractional)')
    plt.title(name)
    plt.suptitle(metadata, fontsize=10)  # Display metadata as subtitle
    plt.legend()
    plt.grid(True)
    
    # Enable interactive tooltips with mplcursors
    cursor = mplcursors.cursor(scatter, hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(
        f'Brightness: {zoomed_brightness[sel.index]:.6f}\nUncertainty: {zoomed_uncertainty[sel.index]:.6f}'))

    plt.show()


if __name__ == "__main__":
    
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Plots the light curve, and optionally plot the transit curve.")
    parser.add_argument("csv_filepath", type=float, help="path to the csv file.")

    parser.add_argument("--start_time", type=float, help="The start time of the transit (optional).")
    parser.add_argument("--end_time", type=float, help="The end time of the transit (optional).")

    # Parse the arguments
    args = parser.parse_args()


    file_path = args.csv_filepath  # Replace with your CSV file path

    time, brightness_fractional, brightness_uncertainty, star_metadata = read_csv(file_path)
    
    peaks_data = find_spikes(time, brightness_fractional)

    plot_light_curve(time, brightness_fractional, brightness_uncertainty, star_metadata, peaks_data, file_path)

    if args.start_time and args.end_time:
        label = input("Add name for transit:")
        plot_transit_curve(time, brightness_fractional, brightness_uncertainty, star_metadata, args.start_time, args.end_time, label)