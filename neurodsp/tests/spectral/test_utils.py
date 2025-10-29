"""Tests for neurodsp.spectral.utils."""

import pytest

import numpy as np
from numpy.testing import assert_equal

from neurodsp.tests.tsettings import FS

from neurodsp.spectral.utils import *

###################################################################################################
###################################################################################################

def test_trim_spectrum():

    freqs = np.array([5, 6, 7, 8, 9])
    pows = np.array([1, 2, 3, 4, 5])

    freqs_new, pows_new = trim_spectrum(freqs, pows, [6, 8])
    assert_equal(freqs_new, np.array([6, 7, 8]))
    assert_equal(pows_new, np.array([2, 3, 4]))

def test_trim_spectrogram():

    freqs = np.array([5, 6, 7, 8])
    times = np.array([0, 1, 2,])
    pows = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9],
                     [10, 11, 12]])

    f_ext, t_ext, p_ext = trim_spectrogram(freqs, times, pows, f_range=[6, 8], t_range=[0, 1])
    assert_equal(f_ext, np.array([6, 7, 8]))
    assert_equal(t_ext, np.array([0, 1]))
    assert_equal(p_ext, np.array([[4, 5], [7, 8], [10, 11]]))

    # Check extraction across specified axis
    f_ext, t_ext, p_ext = trim_spectrogram(freqs, times, pows, f_range=None, t_range=[0, 1])
    assert_equal(f_ext, freqs)
    assert_equal(t_ext, np.array([0, 1]))
    f_ext, t_ext, p_ext = trim_spectrogram(freqs, times, pows, f_range=[6, 8], t_range=None)
    assert_equal(f_ext, np.array([6, 7, 8]))
    assert_equal(t_ext, times)

def test_pad_signal():

    # Test case: odd length, even number added per side
    length = 5
    out1 = pad_signal(np.array([1, 2, 3]), length)
    assert len(out1) == length

    # Test case: even length, even number added per side
    length = 6
    out2 = pad_signal(np.array([1, 2]), length)
    assert len(out2) == length

    # Test case: odd length, uneven number added per side
    length = 5
    out3 = pad_signal(np.array([1, 2]), length)
    assert len(out3) == length

    # Test case: even length, uneven number added per side
    length = 6
    out4 = pad_signal(np.array([1, 2, 3]), length)
    assert len(out4) == length
