"""Tests for neurodsp.plts.rhythm."""

import numpy as np

from neurodsp.tests.tsettings import TEST_PLOTS_PATH
from neurodsp.tests.tutils import plot_test

from neurodsp.plts.rhythm import *

###################################################################################################
###################################################################################################

@plot_test
def test_plot_swm_pattern():

    data = np.arange(0, 2, 0.1)

    plot_swm_pattern(data,
                     save_fig=True, file_path=TEST_PLOTS_PATH,
                     file_name='test_plot_swm_pattern.png')

@plot_test
def test_plot_lagged_coherence():

    data = np.arange(0, 2, 0.1)

    plot_lagged_coherence(data, data,
                          save_fig=True, file_path=TEST_PLOTS_PATH,
                          file_name='test_plot_lagged_coherence.png')
