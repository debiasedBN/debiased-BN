#!/usr/bin/env python3
"""
Description: Downsample and bandpass filter EEG data.
License: MIT
"""

import os
import argparse
import requests
import zipfile
import vaex
import numpy as np
from tqdm import tqdm
from scipy.signal import butter, filtfilt, decimate


CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'FC3', 'FCz', 'FC4', 'T7', 'C3', 'Cz', 'C4',
    'T8', 'CP3', 'CPz', 'CP4', 'P7', 'P3', 'Pz',
    'P4', 'P8', 'O1', 'Oz', 'O2',
]

def download_and_extract_dataset(url, download_path, extract_path, zip_password=None):
    """
    Download and extract a dataset from a given URL.

    Parameters:
        url (str): URL of the dataset zip file.
        download_path (str): Local file path to save the downloaded zip file.
        extract_path (str): Directory path to extract the dataset.
        zip_password (str, optional): Password for the zip file, if required.
    """
    if not os.path.exists(download_path):
        print(f"Downloading dataset from {url} ...")
        response = requests.get(url)
        response.raise_for_status()
        with open(download_path, "wb") as f:
            f.write(response.content)
    os.makedirs(extract_path, exist_ok=True)
    with zipfile.ZipFile(download_path, "r") as zip_ref:
        if zip_password:
            zip_ref.extractall(path=extract_path, pwd=zip_password.encode())
        else:
            zip_ref.extractall(path=extract_path)

def bandpass_and_downsample_eeg_signal(
    eeg_data,
    original_fs=500,
    target_fs=100,
    lower_bound=0.5,
    upper_bound=50,
    filter_order=4,
):
    """
    Apply a Butterworth bandpass filter to EEG data and downsample it.

    Parameters:
        eeg_data (numpy.ndarray): EEG signal array with shape (samples, channels).
        original_fs (int): Original sampling frequency in Hz.
        target_fs (int): Target sampling frequency in Hz.
        lower_bound (float): Lower frequency bound for the bandpass filter.
        upper_bound (float): Upper frequency bound for the bandpass filter.
        filter_order (int): Order of the Butterworth filter.

    Returns:
        numpy.ndarray: Filtered and downsampled EEG data.
    """
    nyquist = original_fs / 2
    low = lower_bound / nyquist
    high = upper_bound / nyquist
    b, a = butter(filter_order, [low, high], btype="band")
    downsample_factor = original_fs // target_fs
    filtered = np.array([
        decimate(filtfilt(b, a, eeg_data[:, i]), downsample_factor, ftype="iir")
        for i in range(eeg_data.shape[1])
    ]).T
    return filtered

def downsample_raw_data(input_directory, output_directory, channels=CHANNELS):
    """
    Process raw EEG CSV files by applying bandpass filtering and downsampling,
    then export the results as Parquet files.

    Parameters:
        input_directory (str): Path to the directory containing raw CSV files.
        output_directory (str): Output directory for filtered and downsampled data.
        channels (list): List of channel names.
    """
    for subid in tqdm(os.listdir(input_directory)):
        for sess in range(1, 4):
            dir_path = os.path.join(input_directory, f"{subid}/ses-{sess}/eeg")
            if os.path.exists(dir_path):
                for file_name in os.listdir(dir_path):
                    if file_name.endswith(".csv"):
                        session_path = os.path.join(dir_path, file_name)
                        dest_path = os.path.join(output_directory, file_name.replace(".csv", ".parquet"))
                        if not os.path.exists(dest_path):
                            v_df = vaex.open(session_path)
                            data = np.array(v_df)[:, :-7]  # assuming last 7 columns are non-EEG data
                            v_df.close()
                            
                            data_ds = bandpass_and_downsample_eeg_signal(
                                data, original_fs=500, target_fs=100,
                                lower_bound=0.5, upper_bound=50, filter_order=4
                            )
                            
                            df = vaex.from_arrays(**{channel: data_ds[:, i] for i, channel in enumerate(channels)})
                            
                            try:
                                if len(df) >= 1000:
                                    df.export_parquet(dest_path, progress=True)
                                    df.close()
                                else:
                                    print(session_path, "insufficient length")
                            except Exception as e:
                                print(session_path, e)
                        else:
                            print(dest_path, "already exists")

def main():
    """
    Parse command-line arguments and execute the EEG downsampling process.
    """
    parser = argparse.ArgumentParser(description="EEG Data Downsampling Block")
    parser.add_argument("--input", required=True, help="Path to input dataset directory")
    parser.add_argument("--output", required=True, help="Output directory for filtered data")
    parser.add_argument("--download_url", help="URL to download dataset zip file")
    parser.add_argument("--download_path", help="Local path to save downloaded zip file")
    parser.add_argument("--extract_path", help="Local path to extract dataset")
    parser.add_argument("--zip_password", help="Password for the zip file", default=None)
    args = parser.parse_args()
    
    if args.download_url and args.download_path and args.extract_path:
        download_and_extract_dataset(args.download_url, args.download_path, args.extract_path, args.zip_password)
    
    os.makedirs(args.output, exist_ok=True)
    downsample_raw_data(args.input, args.output)

if __name__ == "__main__":
    main()
