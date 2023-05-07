import numpy as np
import pytest
import rydiqule as rq


@pytest.mark.exp
def test_calculate_snr():
    """
    Check the SNR calculation agains a basic result from https://arxiv.org/pdf/2105.10494.pdf.
    Show that for a weak off-resonant E-field, in the phase quadrature, with small decoherence,
    and in the limit of small Rabi frequencies, that the optimum probe Rabis is sqrt(2) lower
    than the coupling.  Note that this test does not check the overal scalling of the SNR.
    """

    Rb_sensor = rq.Sensor(4)

    kappa = 28759.120135025692
    eta = 0.0013537502717366691

    red_rabi = np.linspace(0.1,4,100)
    blue_rabi = 1

    rf_rabi_step = np.array([0.1, 0.101])
    probe = {'states': (0,1), 'rabi_frequency': red_rabi, 'detuning': 0, 'label': 'probe'}
    couple = {'states': (1,2), 'rabi_frequency': blue_rabi, 'detuning': 0, 'label':'couple'}
    rf = {'states': (2,3), 'rabi_frequency': rf_rabi_step, 'detuning':30, 'label': 'rf'}

    gamma_matrix = np.array(
                          [[0, 0, 0, 0],
                           [37.82675, 0, 0, 0],
                           [0.02, 0, 0, 0],
                           [0, 0, 0, 0]]
                          )

    Rb_sensor.add_couplings(probe, couple, rf)

    Rb_sensor.set_gamma_matrix(gamma_matrix)

    snrs, param_mesh = rq.get_snr(
        Rb_sensor, 100e-6, param_label='rf_rabi_frequency', probe_tuple=(0,1),
        phase_quadrature=True, kappa=kappa, eta=eta
        )

    snrs_final = snrs[:,1]
    max_idx = np.argmax(snrs_final)
    param_mesh_final = np.array(param_mesh)[:,:,1]

    optimum_rabi_result = param_mesh_final[0,max_idx]
    prediction = blue_rabi/np.sqrt(2)

    assert optimum_rabi_result == pytest.approx(prediction, abs=0, rel=0.1), \
        'SNR with sensor does not match theory approximation'

    # Now check with a Cell to confirm general operation
    r1 = [50, 2, 2.5, 0.5]
    r2 = [51, 1, 1.5, 0.5]

    Rb_cell = rq.Cell('Rb85', *rq.D2_states("Rb85"), r1, r2, gamma_transit=0, beam_area=1e-6)

    Rb_cell.add_couplings(probe, couple, rf)

    snrs, param_mesh = rq.get_snr(
        Rb_cell, 100e-6, param_label='rf_rabi_frequency', probe_tuple=(0,1),
        phase_quadrature=True,
    )

    snrs_final = snrs[:,1]
    max_idx = np.argmax(snrs_final)
    param_mesh_final = np.array(param_mesh)[:,:,1]

    optimum_rabi_result = param_mesh_final[0,max_idx]
    prediction = blue_rabi/np.sqrt(2)

    assert optimum_rabi_result == pytest.approx(prediction, abs=0, rel=0.1), \
        'SNR with cell does not match theory approximation'
