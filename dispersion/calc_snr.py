#!/usr/bin/env python3
"""
SNR calculation executable for gravitational wave signals in next-generation detectors.

This module provides a command-line interface for calculating optimal SNR for binary 
black hole mergers using frequency-dependent antenna responses.
"""

import bilby
import numpy as np
from bilby.core.utils import speed_of_light
from bilby.core.utils.constants import solar_mass, gravitational_constant
import bilby.gw.utils as gwutils
from copy import deepcopy
import h5py
import argparse
import sys
import os
import logging
from gw_analysis import setup_detector_and_waveform_generator, calc_waveform

# Suppress bilby logging
bilby.core.utils.logger.setLevel(logging.ERROR)


def calc_snr(injection_parameters, asd_file, detector_file, 
             minimum_frequency=5, sampling_frequency=4096, reference_frequency=100,
             waveform_approximant="IMRPhenomXPHM", mode_array=None, long_wavelength_approximation=False):
    """
    Calculate the optimal SNR for a gravitational wave signal.
    
    Parameters
    ----------
    injection_parameters : dict
        Dictionary containing injection parameters including:
        - mass_1, mass_2 : float
            Component masses in solar masses
        - ra, dec : float  
            Right ascension and declination in radians
        - theta_jn, psi, phase : float
            Inclination, polarization angle, and phase in radians
        - geocent_time : float
            GPS time of coalescence
        - luminosity_distance : float
            Distance in Mpc
        - chi_1, chi_2 : float, optional
            Dimensionless spin parameters (default: 0)
            
    asd_file : str
        Path to ASD file containing frequency and strain columns
        
    detector_file : str  
        Path to detector configuration file (.ifo format)
        
    minimum_frequency : float, optional
        Minimum frequency for analysis in Hz (default: 5)
        
    sampling_frequency : float, optional
        Sampling frequency in Hz (default: 4096)
        
    reference_frequency : float, optional  
        Reference frequency in Hz (default: 100)
        
    waveform_approximant : str, optional
        Waveform model to use (default: "IMRPhenomXPHM")
        
    mode_array : list, optional
        List of [l,m] mode pairs to include (default: [[2,2]])
        
    long_wavelength_approximation : bool, optional
        Whether to use long wavelength approximation (default: False)
        
    Returns
    -------
    tuple
        (snr, duration, start_time) where:
        - snr : float - Optimal signal-to-noise ratio
        - duration : float - Waveform duration in seconds  
        - start_time : float - GPS start time
    """

    waveform_minimum_frequency = 5

    # Use shared setup function
    ifo, waveform_generator, duration, start_time = setup_detector_and_waveform_generator(
        injection_parameters, asd_file, detector_file, waveform_minimum_frequency, 
        sampling_frequency, reference_frequency, waveform_approximant, mode_array
    )
    
    # Determine xG effects based on duration
    if duration > 15 * 60:
        xG_effects = [True, True, not long_wavelength_approximation]
    else:
        xG_effects = [False, False, not long_wavelength_approximation]
    
    # Use calc_waveform to get the strain
    frequencies, h = calc_waveform(
        ifo, waveform_minimum_frequency, injection_parameters, 
        waveform_generator, sampling_frequency, xG_effects
    )
    
    # Calculate SNR from strain
    ifo.set_strain_data_from_frequency_domain_strain(
        h, start_time=start_time, frequency_array=frequencies
    )
    ifo.minimum_frequency = minimum_frequency
    snr_squared = ifo.optimal_snr_squared(signal=ifo.frequency_domain_strain)
    snr = np.sqrt(snr_squared.real)
    
    return snr, duration, start_time


def calc_snr_batch_from_h5(h5_file, detector_file, asd_file, detector_name, 
                          n_events=1000, long_wavelength_approximation=False,
                          minimum_frequency=5, sampling_frequency=4096, 
                          reference_frequency=100, waveform_approximant="IMRPhenomXPHM",
                          mode_array=None):
    """
    Calculate SNR for multiple events from an h5 file and store results back to the file.
    
    The h5 file is expected to contain datasets with these names:
    m1, m2, ra, dec, psi, luminosity_distance, theta_jn, phase, chi_1, chi_2, geocentric_time
    
    This function will create new datasets in the h5 file:
    - snr_{detector_name}: Array of SNR values
    - duration_{detector_name}: Array of waveform durations (seconds)
    - start_time_{detector_name}: Array of GPS start times
    
    Parameters
    ----------
    h5_file : str
        Path to HDF5 file containing injection parameters  
    detector_file : str
        Path to detector configuration file
    asd_file : str
        Path to ASD file
    detector_name : str
        Name for the detector (e.g., 'H1', 'L1', 'CE40') - used for SNR column name
    n_events : int, optional
        Number of events to process (default: 1000)
    long_wavelength_approximation : bool, optional
        Whether to use long wavelength approximation (default: False)
    minimum_frequency : float, optional
        Minimum frequency in Hz (default: 5)
    sampling_frequency : float, optional
        Sampling frequency in Hz (default: 4096)
    reference_frequency : float, optional
        Reference frequency in Hz (default: 100)
    waveform_approximant : str, optional
        Waveform approximant (default: "IMRPhenomXPHM")
    mode_array : list, optional
        Mode array for waveform generation (default: [[2,2], [3,2], [3,3], [4,4]])
        
    Returns
    -------
    np.ndarray
        Array of calculated SNR values
    """
    
    if mode_array is None:
        mode_array = [[2,2], [3,2], [3,3], [4,4]]
    
    print(f"Processing {n_events} events from {h5_file}")
    print(f"Detector: {detector_name}")
    print(f"Long wavelength approximation: {long_wavelength_approximation}")
    
    # Counter for events with duration > 15 minutes
    long_duration_count = 0
    
    # Open h5 file to read injection parameters
    with h5py.File(h5_file, 'r') as f:
        print(f"Available keys in h5 file: {list(f.keys())}")
        
        # Get total number of events
        total_events = len(f['m1'])
        n_events = min(n_events, total_events)
        print(f"Processing {n_events} out of {total_events} total events")
        
        # Initialize arrays for SNR, duration, and start_time
        snr_values = np.zeros(n_events)
        duration_values = np.zeros(n_events)
        start_time_values = np.zeros(n_events)
        
        # Process each event
        for i in range(n_events):
            if i % 100 == 0:
                print(f"Processing event {i+1}/{n_events}")
            
            # Extract injection parameters for this event
            injection_params = {
                'ra': float(f['ra'][i]),
                'dec': float(f['dec'][i]),
                'theta_jn': float(f['theta_jn'][i]),
                'psi': float(f['psi'][i]),
                'phase': float(f['phase'][i]),
                'geocent_time': float(f['geocentric_time'][i]),
                'luminosity_distance': float(f['luminosity_distance'][i]),
                'chi_1': float(f['chi_1'][i]),
                'chi_2': float(f['chi_2'][i]),
                'chirp_mass': float(f['detector_frame_chirp_mass'][i]),
                'mass_ratio': float(f['q'][i]),
                'a': 0,
                'A': 0
            }
                
            # Calculate SNR, duration, and start_time for this event
            snr, duration, start_time = calc_snr(
                injection_params, asd_file, detector_file,
                minimum_frequency=minimum_frequency,
                sampling_frequency=sampling_frequency,
                reference_frequency=reference_frequency,
                waveform_approximant=waveform_approximant,
                mode_array=mode_array,
                long_wavelength_approximation=long_wavelength_approximation
            )
                
            snr_values[i] = snr
            duration_values[i] = duration
            start_time_values[i] = start_time
            
    # Save SNR, duration, and start_time values back to h5 file
    snr_column_name = f'snr_{detector_name}'
    duration_column_name = f'duration_{detector_name}'
    start_time_column_name = f'start_time_{detector_name}'
    
    print(f"Saving SNR values as '{snr_column_name}' to {h5_file}")
    print(f"Saving duration values as '{duration_column_name}' to {h5_file}")
    print(f"Saving start_time values as '{start_time_column_name}' to {h5_file}")
    
    with h5py.File(h5_file, 'a') as f:
        # Remove existing columns if they exist
        for column_name in [snr_column_name, duration_column_name, start_time_column_name]:
            if column_name in f:
                del f[column_name]
        
        # Create new datasets with calculated values
        # Pad with NaN if n_events < total_events
        if n_events < total_events:
            # SNR values
            full_snr_array = np.full(total_events, np.nan)
            full_snr_array[:n_events] = snr_values
            f.create_dataset(snr_column_name, data=full_snr_array)
            
            # Duration values
            full_duration_array = np.full(total_events, np.nan)
            full_duration_array[:n_events] = duration_values
            f.create_dataset(duration_column_name, data=full_duration_array)
            
            # Start time values
            full_start_time_array = np.full(total_events, np.nan)
            full_start_time_array[:n_events] = start_time_values
            f.create_dataset(start_time_column_name, data=full_start_time_array)
        else:
            f.create_dataset(snr_column_name, data=snr_values)
            f.create_dataset(duration_column_name, data=duration_values)
            f.create_dataset(start_time_column_name, data=start_time_values)
    
    print(f"Completed SNR calculation. Statistics:")
    print(f"  Mean SNR: {np.nanmean(snr_values):.2f}")
    print(f"  Median SNR: {np.nanmedian(snr_values):.2f}")
    print(f"  Max SNR: {np.nanmax(snr_values):.2f}")
    print(f"  Min SNR: {np.nanmin(snr_values):.2f}")
    print(f"  Failed calculations: {np.sum(np.isnan(snr_values))}")
    print(f"  Events with duration > 15 minutes: {long_duration_count}")
    
    return snr_values


def main():
    """Command line interface for SNR calculation."""
    parser = argparse.ArgumentParser(
        description='Calculate SNR for gravitational wave events from h5 file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('h5_file', 
                       help='Path to HDF5 file containing injection parameters (default: bbh_samples.h5)',
                       nargs='?', default='bbh_samples.h5')
    
    parser.add_argument('--detector-file', '-d', required=True,
                       help='Path to detector configuration file (.ifo)')
    
    parser.add_argument('--asd-file', '-a', required=True,
                       help='Path to ASD file containing strain data')
    
    parser.add_argument('--name', '-n', required=True,
                       help='Detector name for SNR column (e.g., H1, L1, CE40)')
    
    parser.add_argument('--n-events', '-N', type=int, default=1000,
                       help='Number of events to process')
    
    parser.add_argument('--long-wavelength', '-l', action='store_true',
                       help='Use long wavelength approximation')
    
    parser.add_argument('--minimum-frequency', type=float, default=5,
                       help='Minimum frequency in Hz')
    
    parser.add_argument('--sampling-frequency', type=float, default=4096,
                       help='Sampling frequency in Hz')
    
    parser.add_argument('--reference-frequency', type=float, default=100,
                       help='Reference frequency in Hz')
    
    parser.add_argument('--waveform-approximant', default="IMRPhenomXPHM",
                       help='Waveform approximant')
    
    parser.add_argument('--mode-array', nargs='+', default=['2,2'],
                       help='Mode array as space-separated l,m pairs (e.g., "2,2" "3,2" "3,3")')
    
    args = parser.parse_args()
    
    # Parse mode array
    mode_array = []
    for mode_str in args.mode_array:
        l, m = map(int, mode_str.split(','))
        mode_array.append([l, m])
    print (f"Using mode array: {mode_array}")
    
    # Check if files exist
    if not os.path.exists(args.h5_file):
        print(f"Error: H5 file not found: {args.h5_file}")
        sys.exit(1)
    
    if not os.path.exists(args.detector_file):
        print(f"Error: Detector file not found: {args.detector_file}")
        sys.exit(1)
        
    if not os.path.exists(args.asd_file):
        print(f"Error: ASD file not found: {args.asd_file}")
        sys.exit(1)
    
    print("=" * 60)
    print("SNR Calculation for Gravitational Wave Events")
    print("=" * 60)
    print(f"H5 file: {args.h5_file}")
    print(f"Detector file: {args.detector_file}")
    print(f"ASD file: {args.asd_file}")
    print(f"Detector name: {args.name}")
    print(f"Number of events: {args.n_events}")
    print(f"Long wavelength approximation: {args.long_wavelength}")
    print(f"Mode array: {mode_array}")
    print("=" * 60)
    
    # Calculate SNRs
    bilby.core.utils.logger.disabled = True
    snr_values = calc_snr_batch_from_h5(
        h5_file=args.h5_file,
        detector_file=args.detector_file,
        asd_file=args.asd_file,
        detector_name=args.name,
        n_events=args.n_events,
        long_wavelength_approximation=args.long_wavelength,
        minimum_frequency=args.minimum_frequency,
        sampling_frequency=args.sampling_frequency,
        reference_frequency=args.reference_frequency,
        waveform_approximant=args.waveform_approximant,
        mode_array=mode_array
    )
        
    print("=" * 60)
    print("SNR calculation completed successfully!")
    print(f"Results saved as 'snr_{args.name}', 'duration_{args.name}', and 'start_time_{args.name}' in {args.h5_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()