"""
EE-473 - Final Project

"A simple Impletentation of a LPC Vocoder"

Bogazici University - January 2022 - Sezai Burak Kantarcı - 2020701087
"""

# In the project report, here are to do's with their headlines.

# TODO: First you should explain Linear Prediction Model for speech production. (1. Introduction & 2. Linear Predictive Coding)
# DONE
# TODO: Second, you should explain what pitch is and how it can be calculated. (3. Pitch and Decision)
# DONE
# TODO: You should also explain how the voiced and unvoiced decision is given during this process. (3. Pitch and Decision)
# DONE
# TODO: Finally you should draw a block diagram of the LPC encoder and decoder. (4. Methodology and Impletentation)
# DONE
# TODO: You should explain your algorithm details. (5. Pipeline)
# DONE
# TODO: You should also provide a sample input and output waveform of 1-2 seconds together with your report. (6. Results)
# DONE

"""

README

In order to run the code, you can simply make an Anaconda Environment; import the packages down below.
In a conda cmd, run:

$ conda install -c anaconda numpy
$ conda install -c anaconda scipy

This script was succesfully tested on Python 3.9.7 kernel.

Here is the output for "$ conda list" command the for used enviroment.

You can use this list to clone the enviroment. =)


backcall                  0.2.0                      py_0    anaconda
blas                      1.0                         mkl
bottleneck                1.3.2            py39h7cc1a96_1
brotli                    1.0.9                h8ffe710_6    conda-forge
brotli-bin                1.0.9                h8ffe710_6    conda-forge
ca-certificates           2020.10.14                    0    anaconda
certifi                   2021.10.8        py39haa95532_2
colorama                  0.4.4                      py_0    anaconda
cycler                    0.11.0             pyhd8ed1ab_0    conda-forge
debugpy                   1.5.1                    pypi_0    pypi
decorator                 4.4.2                      py_0    anaconda
entrypoints               0.3                      pypi_0    pypi
fonttools                 4.28.5           py39hb82d6ee_0    conda-forge
freetype                  2.10.4               h546665d_1    conda-forge
icc_rt                    2019.0.0             h0cc432a_1
icu                       68.2                 h0e60522_0    conda-forge
intel-openmp              2021.4.0          haa95532_3556
ipykernel                 6.7.0                    pypi_0    pypi
ipython                   7.29.0           py39hd4e2768_0
jbig                      2.1               h8d14728_2003    conda-forge
jedi                      0.18.0           py39haa95532_1
jpeg                      9d                   h8ffe710_0    conda-forge
jupyter-client            7.1.1                    pypi_0    pypi
jupyter-core              4.9.1                    pypi_0    pypi
kiwisolver                1.3.2            py39h2e07f2f_1    conda-forge
lcms2                     2.12                 h2a16943_0    conda-forge
lerc                      3.0                  h0e60522_0    conda-forge
libbrotlicommon           1.0.9                h8ffe710_6    conda-forge
libbrotlidec              1.0.9                h8ffe710_6    conda-forge
libbrotlienc              1.0.9                h8ffe710_6    conda-forge
libclang                  11.1.0          default_h5c34c98_1    conda-forge
libdeflate                1.8                  h8ffe710_0    conda-forge
libpng                    1.6.37               h1d00b33_2    conda-forge
libtiff                   4.3.0                hd413186_2    conda-forge
libzlib                   1.2.11            h8ffe710_1013    conda-forge
lz4-c                     1.9.3                h8ffe710_1    conda-forge
matplotlib                3.5.1            py39hcbf5309_0    conda-forge
matplotlib-base           3.5.1            py39h581301d_0    conda-forge
matplotlib-inline         0.1.2              pyhd3eb1b0_2
mkl                       2021.4.0           haa95532_640
mkl-service               2.4.0            py39h2bbff1b_0
mkl_fft                   1.3.1            py39h277e83a_0
mkl_random                1.2.2            py39hf11a4ad_0
munkres                   1.1.4              pyh9f0ad1d_0    conda-forge
nest-asyncio              1.5.4                    pypi_0    pypi
numexpr                   2.8.1            py39hb80d3ca_0
numpy                     1.21.2           py39hfca59bb_0
numpy-base                1.21.2           py39h0829f74_0
olefile                   0.46               pyh9f0ad1d_1    conda-forge
openjpeg                  2.4.0                hb211442_1    conda-forge
openssl                   1.1.1m               h2bbff1b_0
packaging                 21.3               pyhd3eb1b0_0
pandas                    1.3.5            py39h6214cd6_0
parso                     0.8.0                      py_0    anaconda
pickleshare               0.7.5           pyhd3eb1b0_1003
pillow                    8.4.0            py39h916092e_0    conda-forge
pip                       21.2.4           py39haa95532_0
prompt-toolkit            3.0.8                      py_0    anaconda
pygments                  2.7.1                      py_0    anaconda
pyparsing                 3.0.4              pyhd3eb1b0_0
pyqt                      5.12.3           py39hcbf5309_8    conda-forge
pyqt-impl                 5.12.3           py39h415ef7b_8    conda-forge
pyqt5-sip                 4.19.18          py39h415ef7b_8    conda-forge
pyqtchart                 5.12             py39h415ef7b_8    conda-forge
pyqtwebengine             5.12.1           py39h415ef7b_8    conda-forge
python                    3.9.7                h6244533_1
python-dateutil           2.8.2              pyhd3eb1b0_0
python_abi                3.9                      2_cp39    conda-forge
pytz                      2021.3             pyhd3eb1b0_0
pywin32                   303                      pypi_0    pypi
pyzmq                     22.3.0                   pypi_0    pypi
qt                        5.12.9               h5909a2a_4    conda-forge
scipy                     1.7.3            py39h0a974cb_0
setuptools                58.0.4           py39haa95532_0
six                       1.16.0             pyhd3eb1b0_0
sqlite                    3.37.0               h2bbff1b_0
tk                        8.6.11               h8ffe710_1    conda-forge
tornado                   6.1              py39hb82d6ee_2    conda-forge
traitlets                 5.1.1              pyhd3eb1b0_0
tzdata                    2021e                hda174b7_0
vc                        14.2                 h21ff451_1
vs2015_runtime            14.27.29016          h5e58377_2
wcwidth                   0.2.5                      py_0    anaconda
wheel                     0.37.0             pyhd3eb1b0_1
wincertstore              0.2              py39haa95532_2
xz                        5.2.5                h62dcd97_1    conda-forge
zlib                      1.2.11            h8ffe710_1013    conda-forge
zstd                      1.5.1                h6255e5f_0    conda-forge

"""


import numpy as np
import scipy, scipy.io, scipy.io.wavfile, scipy.signal
from numpy.polynomial import polynomial as P
import logging

Log_Format = "%(levelname)s %(asctime)s - %(message)s"

logging.basicConfig(filename = "logfile.log",
                    filemode = "w",
                    format = Log_Format, 
                    level = logging.INFO)


def load_wave(fname):
    try:
        """
        Takes the filename as a string, to load the wav file in workspace.
        Return normalised version of the it.
        """
        sr, wave = scipy.io.wavfile.read(fname)
        
        return wave/32768.0
    except BaseException:
        logging.exception("Problem occured in load_wave")

def pulse_train(samples, frequency, sampling_rate, k):
    try: 
        """
        Pulse Train for voiced speech synthesis.
        samples: number of samples to generate
        Takes base frequency, sampling rate as hertz and k modulation degree
        Returns the train with #of samples wanted.
        """
        # Same sized numpy array as the original signal
        train = np.arange(samples)
        # phase of each element
        phase = (frequency*2*np.pi * (train/float(sampling_rate)))
        # Pulses
        pulses = np.cos(phase) * np.exp(np.cos(phase)* k -(k))    
        
        return pulses

    except BaseException:
        logging.exception("Problem occured in pulse_train")

def linear_predictive_coding(wave, order):
    try:
        """
        Compute LPC of the waveform. 
        a: the LPC coefficients
        e: the total error
        k: the reflection coefficients
        """    
        # Take the autocorrelation of the given waveform.
        autocorr = scipy.signal.correlate(wave, wave)[len(wave)-1:]/len(wave)
        # Calculate the coefficients, error and reflection coefficients. 
        a, e, k  = levinson_one_dimensional(autocorr, order)

        return a,e,k

    except BaseException:
        logging.exception("Problem occured in linear_predictive_coding")

def linear_predictive_coding_to_line_spectral_pairs(line_spectral_pairs):    
    try:
        """
        Convert Linear Predictive Coding Coefficents to Line Spectral Pairs
        Returns Line Spectral Pairs
        """
        lenght = len(line_spectral_pairs)+1
        lpc_coefficients = np.zeros((lenght,))        

        lpc_coefficients[0:-1] = line_spectral_pairs

        # Allocate arrays
        p = np.zeros((lenght,))
        q = np.zeros((lenght,))    

        for i in range(lenght):
            j = lenght-i-1
            p[i] = lpc_coefficients[i] + lpc_coefficients[j]
            q[i] = lpc_coefficients[i] - lpc_coefficients[j]

        # Return the angles of the roots of p and q; sorted
        ps = np.sort(np.angle(np.roots(p)))
        qs = np.sort(np.angle(np.roots(q)))

        # Stack the array in sequence vertically
        lsp = np.vstack([ps[:len(ps)//2],qs[:len(qs)//2]]).T

        return lsp

    except BaseException:
        logging.exception("Problem occured in linear_predictive_coding_to_line_spectral_pairs")

def line_spectral_pairs_to_linear_predictive_coding(line_spectral_pairs):  
    try:
        """
        Takes line spectral pairs
        Returns Line Prediction Coding coefficients
        Basically compansating the process in lpc_to_lsp
        """
        ps = np.concatenate((line_spectral_pairs[:,0], -line_spectral_pairs[::-1,0], [np.pi]))
        qs = np.concatenate((line_spectral_pairs[:,1], [0], -line_spectral_pairs[::-1,1]))

        p = np.cos(ps) - np.sin(ps)*1.0j
        q = np.cos(qs) - np.sin(qs)*1.0j

        p = np.real(P.polyfromroots(p))
        q = -np.real(P.polyfromroots(q))

        a = 0.5 * (p+q)
        
        return a[:-1]
        
    except BaseException:
        logging.exception("Problem occured in line_spectral_pairs_to_linear_predictive_coding")

def levinson_one_dimensional(r, order):
    try:
        """
        Using Levinson-Durbin Recursion in order to find the coefficients of a length(r)-1 order autoregressive linear process
        r is the input autocorrelation vector, order is the LPC order.
        Returns regression_coefficients, errors, reflection_coefficients

        """
        # To process the matrix, make it a one dimensional array
        r = np.atleast_1d(r)

        # Estimated coefficients
        regression_coefficients = np.empty(order+1, r.dtype)

        # temporary array, in order to recrusively change the coefficients at every step
        temp = np.empty(order+1, r.dtype)

        # Reflection coefficients
        reflection_coefficients = np.empty(order, r.dtype)

        # Lag is zero, first element is "R1"
        regression_coefficients[0] = 1.

        # Prediction Errors
        errors = r[0]

        # At each iteration, calculate autoregressive coefficients, their multiplication with the previos step, and error.
        for i in range(1, order+1):
            acc = r[i]
            for j in range(1, i):
                acc += regression_coefficients[j] * r[i-j]
            reflection_coefficients[i-1] = -acc / errors
            regression_coefficients[i] = reflection_coefficients[i-1]

            for j in range(order):
                temp[j] = regression_coefficients[j]

            for j in range(1, i):
                regression_coefficients[j] += reflection_coefficients[i-1] * np.conj(temp[i-j])

            errors *= 1 - reflection_coefficients[i-1] * np.conj(reflection_coefficients[i-1])

        # return the coefficients, errors and reflection coefficients
        return regression_coefficients, errors, reflection_coefficients
        
    except BaseException:
        logging.exception("Problem occured in levinson_one_dimensional")

def lpc_vocode(wave, window_size, order, carrier, residual_amp=0.0, 
                vocode_amp=1.0, freq_shift=1.0):    
    try:
        """
        Apply LPC vocoding to a pair of signals.

        Takes the modulator wave, window length, LPC order, carrier wave,
        residual amplitute, vocoded amplitude.
        Takes envs to control volume modulation, and freq_shift to control
        frequency shifting weight. 

        Returns Vocoded singal.
        """
        # Hamming window
        window = scipy.signal.hann(window_size)

        #allocate the array for the output
        vocode = np.zeros(len(wave+window_size))    

        # 50% window steps for overlap-add
        for i in range(0,len(wave),window_size//2):

            # slice the wave and the carrier
            wave_slice = wave[i:i+window_size]
            carrier_slice = carrier[i:i+window_size]

            if len(wave_slice)==window_size:                        
                # compute LPC
                lpc_coefficients,error,reflection = linear_predictive_coding(wave_slice, order)           

                # Getting into the Line Spectral Pairs is easier to make the frequency shifting.
                lsp = linear_predictive_coding_to_line_spectral_pairs(lpc_coefficients)

                # Shift the Line Spectral Pairs
                lsp = (lsp * freq_shift+np.pi) % (np.pi) -np.pi     

                # Get back to the Linear Predictive Coding Coefficents
                lpc_coefficients = line_spectral_pairs_to_linear_predictive_coding(lsp)

                # compute the LPC residual         
                residual = scipy.signal.lfilter(lpc_coefficients, 1., wave_slice)           
                
                # filter, using LPC as the *IIR* component         
                vocoded = scipy.signal.lfilter([1.], lpc_coefficients, carrier_slice)             

                # match Root Mean Square of original signal
                voc_amp = 1e-5 + np.sqrt(np.mean(vocoded**2))
                wave_amp = 1e-5 + np.sqrt(np.mean(wave_slice**2))
                
                # Weight the result
                vocoded = vocoded * (wave_amp/voc_amp)

                # Hann window 50%-overlap-add to remove clicking
                vocode[i:i+window_size] +=  (vocoded * vocode_amp + residual * residual_amp) * window

        return vocode[:len(wave)]

    except BaseException:
        logging.exception("Problem occured in lpc_vocode")

def main():

    logging.info("Started Main Function")
    modulator = load_wave("input/samplespeech.wav")

    # A pulse train with exponentially decreasing modulation degree
    carrier = pulse_train(len(modulator), frequency=40*np.floor(np.linspace(1,6,len(modulator)))**0.25,
                         sampling_rate=44100, k=10**np.linspace(4,2,len(modulator)))
    logging.info("Succesfully made carrier signal")

    # Constructing the vocoded signal from the original and the carrier.
    vocoded = lpc_vocode(modulator, window_size=1000, order=10, carrier=carrier, 
                         residual_amp=0.3, vocode_amp=1, freq_shift=1)
    logging.info("Succesfully made vocoded signal")

    # Original Signal
    scipy.io.wavfile.write("output/original.wav", 44100, modulator.T)
    logging.info("Saved Original signal")

    # Carrier Signal
    scipy.io.wavfile.write("output/carrier.wav", 44100, carrier.T)   
    logging.info("Saved Carrier signal")

    # Vocoded Signal
    scipy.io.wavfile.write("output/vocoded.wav", 44100, vocoded.T)
    logging.info("Saved Vocoded signal")

    logging.info("Made it to the end! =)") 

if __name__ == '__main__':
    main()