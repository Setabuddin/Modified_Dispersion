#!/usr/bin/env python3
"""
Gravitational wave analysis module for next-generation detectors.

This module provides comprehensive functionality for calculating waveforms, SNR, 
and Fisher information matrices for binary black hole mergers using frequency-dependent 
antenna responses.
"""

import bilby
import numpy as np
from bilby.core.utils import speed_of_light
from bilby.core.utils.constants import solar_mass, gravitational_constant
import bilby.gw.utils as gwutils
from copy import deepcopy
import logging

# Suppress bilby logging
bilby.core.utils.logger.setLevel(logging.ERROR)


def setup_detector_and_waveform_generator(injection_parameters, asd_file=None, detector_file=None, 
                                          minimum_frequency=5, sampling_frequency=4096, 
                                          reference_frequency=100, waveform_approximant="IMRPhenomXPHM",
                                          mode_array=None, ifo=None):
    """
    Set up detector and waveform generator for gravitational wave analysis.
    
    Parameters
    ----------
    injection_parameters : dict
        Injection parameters containing masses and other source properties
    asd_file : str, optional
        Path to ASD file (required if ifo not provided)
    detector_file : str, optional  
        Path to detector file (required if ifo not provided)
    minimum_frequency : float, optional
        Minimum frequency in Hz (default: 5)
    sampling_frequency : float, optional
        Sampling frequency in Hz (default: 4096)
    reference_frequency : float, optional
        Reference frequency in Hz (default: 100)
    waveform_approximant : str, optional
        Waveform approximant (default: "IMRPhenomXPHM")
    mode_array : list, optional
        Mode array (default: [[2,2], [3,2], [3,3], [4,4]])
    ifo : bilby.gw.detector.Interferometer, optional
        Pre-configured interferometer (if None, will create from files)
        
    Returns
    -------
    tuple
        (ifo, waveform_generator, duration, start_time)
    """
    if mode_array is None:
        mode_array = [[2,2], [3,2], [3,3], [4,4]]
    
    
    # Use pre-calculated detector_frame_chirp_mass from h5 file
    detector_frame_chirp_mass = injection_parameters['chirp_mass']
    
    # Calculate waveform duration based on chirp mass
    tc_offset = 1
    chirp_mass_in_seconds = detector_frame_chirp_mass * solar_mass * gravitational_constant / speed_of_light**3
    t0 = -5. / 256. * chirp_mass_in_seconds * (np.pi * chirp_mass_in_seconds * minimum_frequency)**(-8. / 3.)
    pow = (np.int32(np.log2(np.abs(t0)))+1)
    if pow < 0:
        pow = 1
    duration = 2**pow
    start_time = injection_parameters["geocent_time"] - duration + tc_offset
    
    # Set up interferometer if not provided
    if ifo is None:
        if asd_file is None or detector_file is None:
            raise ValueError("Must provide either ifo or both asd_file and detector_file")
        
        # Load ASD data
        frequencies_asd, strain_asd = np.loadtxt(asd_file, unpack=True)
        
        # Set up interferometer
        ifo = bilby.gw.detector.load_interferometer(detector_file)
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=frequencies_asd,
            asd_array=strain_asd
        )
    
    # Set up waveform generator
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration, 
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole_individual_modes,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(
            waveform_approximant=waveform_approximant, 
            reference_frequency=reference_frequency, 
            minimum_frequency=minimum_frequency, 
            mode_array=mode_array
        )
    )
    
    return ifo, waveform_generator, duration, start_time


def calc_waveform(ifo, minimum_frequency, injection_parameters, waveform_generator, 
                 sampling_frequency=4096, xG_effects=None):
    """
    Calculate waveform for a gravitational wave signal.
    
    Parameters
    ----------
    ifo : bilby.gw.detector.Interferometer
        Detector object
    minimum_frequency : float
        Minimum frequency for analysis in Hz
    injection_parameters : dict
        Injection parameters
    waveform_generator : bilby.gw.WaveformGenerator
        Waveform generator object
    sampling_frequency : float, optional
        Sampling frequency in Hz (default: 4096)
    xG_effects : list, optional
        List of xG effects [earth_rotation_time_delay, earth_rotation_beam_patterns, finite_size]
        If None, will be determined automatically based on duration
        
    Returns
    -------
    tuple
        (frequencies, h) where frequencies is frequency array, h is strain
    """
    if xG_effects is None:
        xG_effects = [True, True, True]
    
    # Calculate duration and start time using shared logic
    luminosity_distance = injection_parameters['luminosity_distance']
    
    # Use chirp_mass from converted parameters
    detector_frame_chirp_mass = injection_parameters['chirp_mass']
    
    tc_offset = 1
    chirp_mass_in_seconds = detector_frame_chirp_mass * solar_mass * gravitational_constant / speed_of_light**3
    t0 = -5. / 256. * chirp_mass_in_seconds * (np.pi * chirp_mass_in_seconds * minimum_frequency)**(-8. / 3.)
    pow = (np.int32(np.log2(np.abs(t0)))+1)
    if pow < 0:
        pow = 1
    duration = 2**pow
    geocent_time = injection_parameters['geocent_time']
    start_time = geocent_time - duration + tc_offset

    # Determine earth rotation effects based on duration
    if duration > 15 * 60:
        earth_rotation_time_delay = True
        earth_rotation_beam_patterns = True
    else:
        earth_rotation_time_delay = False
        earth_rotation_beam_patterns = False
    
    # Override with user-specified xG_effects if provided
    if xG_effects is not None:
        earth_rotation_time_delay = xG_effects[0] if len(xG_effects) > 0 else earth_rotation_time_delay
        earth_rotation_beam_patterns = xG_effects[1] if len(xG_effects) > 1 else earth_rotation_beam_patterns
        finite_size = xG_effects[2] if len(xG_effects) > 2 else True
    else:
        finite_size = True

    frequencies = waveform_generator.frequency_array
    idxs_above_minimum_frequency = frequencies > (minimum_frequency - (frequencies[1] - frequencies[0]))
    freqs = frequencies[idxs_above_minimum_frequency]

    converted_injection_parameters, _ = waveform_generator.parameter_conversion(injection_parameters)
    waveform_polarizations = waveform_generator.frequency_domain_strain(converted_injection_parameters)
    waveform_polarizations_reduced = {}
    
    # Handle both single and multi-mode waveforms
    if 'plus' in waveform_polarizations.keys():
        waveform_polarizations_reduced['plus'] = waveform_polarizations['plus'][idxs_above_minimum_frequency]
        waveform_polarizations_reduced['cross'] = waveform_polarizations['cross'][idxs_above_minimum_frequency]
    else:
        for key in waveform_polarizations.keys():
            waveform_polarizations_reduced[key] = {}
            waveform_polarizations_reduced[key]['plus'] = waveform_polarizations[key]['plus'][idxs_above_minimum_frequency]
            waveform_polarizations_reduced[key]['cross'] = waveform_polarizations[key]['cross'][idxs_above_minimum_frequency]

    h = np.zeros_like(waveform_generator.frequency_array, dtype=complex)
    h[idxs_above_minimum_frequency] = ifo.get_detector_response_for_frequency_dependent_antenna_response(
        waveform_polarizations=waveform_polarizations_reduced,
        parameters=converted_injection_parameters,
        start_time=start_time,
        frequencies=freqs,
        earth_rotation_time_delay=earth_rotation_time_delay,
        earth_rotation_beam_patterns=earth_rotation_beam_patterns,
        finite_size=finite_size,)
        
    return frequencies, h






def calc_waveform_derivatives(h, ifo, minimum_frequency, injection_parameters, waveform_generator, xG_effects, key, e, sampling_frequency=4096):
    """Calculate numerical derivatives of waveform with respect to parameters."""
    injection_parameters_e = deepcopy(injection_parameters)
    injection_parameters_e[key] += e[key]
    _, h_e = calc_waveform(ifo, minimum_frequency, injection_parameters_e, waveform_generator, sampling_frequency, xG_effects)
    return (h_e-h)/e[key]


def check_derivative_stability(h, ifo, minimum_frequency, injection_parameters, waveform_generator, xG_effects, sampling_frequency=4096, array=None):
    """Check stability of numerical derivatives for different step sizes."""
    import matplotlib.pyplot as plt
    if array is None:
        array = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]

    num_keys = len(injection_parameters.keys())
    rows = (num_keys + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(20, 4 * rows))

    if rows == 1:
        axes = axes.reshape(-1)

    axes = axes.flatten() if num_keys > 1 else [axes]
    for idx, key in enumerate(injection_parameters.keys()):
        ax = axes[idx] if num_keys > 1 else axes
        derivatives = []
        for e_ in array:
            t = calc_waveform_derivatives(h, ifo, minimum_frequency, injection_parameters, waveform_generator, xG_effects, key, {key: e_}, sampling_frequency)
            derivatives.append(np.abs(np.vdot(t, t)))
        ax.plot(array, np.abs(derivatives))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Step Size")
        ax.set_ylabel("Derivative Magnitude")
        ax.set_title(f"Derivative w.r.t. {key}")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Hide any unused axes
    for idx in range(num_keys, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()


def remove(matrix, idx):
    """Remove the row and column of a matrix given an index."""
    return np.delete(np.delete(matrix, idx, axis=0), idx, axis=1)


def calc_Fisher(h, frequencies, ifo, minimum_frequency, injection_parameters, waveform_generator, xG_effects, e, sampling_frequency=4096, skip=[]):
    """Calculate Fisher information matrix for parameter estimation."""
    Fisher = np.zeros((len(injection_parameters), len(injection_parameters)), dtype=float)
    h_derivatives = {}
    
    for i, key in enumerate(injection_parameters.keys()):
        if key in skip:
            continue
        h_derivatives[key] = calc_waveform_derivatives(h, ifo, minimum_frequency, injection_parameters, waveform_generator, xG_effects, key, e, sampling_frequency)/ifo.amplitude_spectral_density_array
    
    for i, key in enumerate(injection_parameters.keys()):
        if key in skip:
            continue
        for j, key2 in enumerate(injection_parameters.keys()):
            if key2 in skip:
                continue
            if i<=j:
                Fisher[i, j] = 4*np.real(np.vdot(h_derivatives[key], h_derivatives[key2])) * (frequencies[1]-frequencies[0])
            else:
                Fisher[i, j] = Fisher[j, i]
        indx = []
    Fisher_tmp = Fisher.copy()
    for i in range(Fisher_tmp.shape[0]):
        if np.sum(Fisher_tmp[i]) == 0:
            indx.append(i)
    Fisher_tmp = remove(Fisher_tmp, indx)
    return Fisher_tmp


def get_samples_from_Fisher(Fisher, parameters, n_samples=10000):
    """Generate parameter samples from Fisher matrix covariance."""
    cov = np.linalg.pinv(Fisher, hermitian=True)
    mean = deepcopy(parameters)
    samples = dict(zip(mean.keys(), np.transpose(np.random.multivariate_normal(list(mean.values()), cov, n_samples))))
    import pandas as pd
    return pd.DataFrame(samples)


