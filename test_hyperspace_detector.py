#!/usr/bin/env python3
"""
Test suite for Hyperspace Wave Detection System

Validates that the detector can:
1. Generate hyperspace wave signals correctly
2. Extract psi signatures accurately
3. Distinguish hyperspace waves from EM waves
4. Distinguish hyperspace waves from noise
5. Meet all detection criteria
"""

import numpy as np
import sys


def test_transmitter_generation():
    """Test that transmitter generates correct signal form."""
    print("Test 1: Transmitter Signal Generation")
    print("-" * 50)
    
    from hyperspace_wave_detector import HyperspaceWaveTransmitter
    
    tx = HyperspaceWaveTransmitter(carrier_freq=1e9, psi_freq=1e6)
    
    # Generate test signal
    t = np.linspace(0, 1e-6, 1000)
    psi = np.linspace(0, 1e-6, 1000)
    
    signal = tx.generate_signal(t, psi)
    
    # Verify properties
    assert len(signal) == len(t), "Signal length mismatch"
    assert np.all(np.isfinite(signal)), "Signal contains non-finite values"
    assert signal.dtype == np.complex128, "Signal should be complex"
    
    # Check that signal has both real-time oscillation and psi modulation
    amplitude = np.abs(signal)
    assert amplitude[0] > amplitude[-1], "Amplitude should decrease with psi"
    
    wavelength = tx.get_wavelength()
    assert 0.1 < wavelength < 1.0, f"Wavelength {wavelength} out of expected range"
    
    psi_scale = tx.get_psi_scale()
    assert psi_scale > 0, "Psi scale should be positive"
    
    print("✓ Transmitter generates valid signals")
    print(f"  Wavelength: {wavelength*100:.2f} cm")
    print(f"  Psi scale: {psi_scale*1e6:.2f} μs")
    print()
    return True


def test_receiver_signature_extraction():
    """Test that receiver can extract psi signatures."""
    print("Test 2: Psi Signature Extraction")
    print("-" * 50)
    
    from hyperspace_wave_detector import (
        HyperspaceWaveTransmitter,
        HyperspaceWaveReceiver
    )
    
    tx = HyperspaceWaveTransmitter(carrier_freq=1e9, psi_freq=1e6)
    rx = HyperspaceWaveReceiver(sampling_rate=1e10)
    
    # Generate clean hyperspace signal
    t = np.linspace(0, 1e-6, 10000)
    psi = np.linspace(0, 5e-6, 10000)
    
    signal = tx.generate_signal(t, psi)
    
    # Extract signature (no noise)
    psi_profile, coherence = rx.extract_psi_signature(signal, t)
    
    assert len(psi_profile) == len(t), "Profile length mismatch"
    assert 0 <= coherence <= 1, f"Coherence {coherence} out of bounds"
    
    # For clean hyperspace signal, coherence should be very high
    assert coherence > 0.7, f"Clean signal coherence {coherence} too low"
    
    print("✓ Receiver extracts psi signatures correctly")
    print(f"  Clean signal coherence: {coherence:.4f}")
    print()
    return True


def test_hyperspace_vs_em_distinction():
    """Test that hyperspace waves are distinguished from EM waves."""
    print("Test 3: Hyperspace vs EM Wave Distinction")
    print("-" * 50)
    
    from hyperspace_wave_detector import (
        HyperspaceWaveTransmitter,
        HyperspaceWaveReceiver
    )
    
    tx = HyperspaceWaveTransmitter(carrier_freq=1e9, psi_freq=1e6)
    rx = HyperspaceWaveReceiver(sampling_rate=1e10)
    
    t = np.linspace(0, 1e-6, 10000)
    
    # Hyperspace signal with psi modulation
    psi = np.linspace(0, 5e-6, 10000)
    hs_signal = tx.generate_signal(t, psi)
    _, hs_coherence = rx.extract_psi_signature(hs_signal, t)
    
    # EM signal (no psi modulation)
    em_signal = tx.generate_signal(t, np.zeros_like(t))
    _, em_coherence = rx.extract_psi_signature(em_signal, t)
    
    # Hyperspace should have much higher coherence
    ratio = hs_coherence / (em_coherence + 0.01)
    
    assert hs_coherence > em_coherence, "Hyperspace should have higher coherence than EM"
    assert ratio > 2.0, f"Coherence ratio {ratio:.2f} too low for distinction"
    
    print("✓ Hyperspace waves distinguished from EM waves")
    print(f"  Hyperspace coherence: {hs_coherence:.4f}")
    print(f"  EM coherence: {em_coherence:.4f}")
    print(f"  Ratio: {ratio:.2f}x")
    print()
    return True


def test_hyperspace_vs_noise_distinction():
    """Test that hyperspace waves are distinguished from noise."""
    print("Test 4: Hyperspace vs Noise Distinction")
    print("-" * 50)
    
    from hyperspace_wave_detector import (
        HyperspaceWaveTransmitter,
        HyperspaceWaveReceiver
    )
    
    tx = HyperspaceWaveTransmitter(carrier_freq=1e9, psi_freq=1e6)
    rx = HyperspaceWaveReceiver(sampling_rate=1e10)
    
    t = np.linspace(0, 1e-6, 10000)
    
    # Hyperspace signal
    psi = np.linspace(0, 5e-6, 10000)
    hs_signal = tx.generate_signal(t, psi)
    _, hs_coherence = rx.extract_psi_signature(hs_signal, t)
    
    # Pure noise
    noise = np.random.randn(len(t)) + 1j * np.random.randn(len(t))
    _, noise_coherence = rx.extract_psi_signature(noise, t)
    
    # Hyperspace should have much higher coherence
    ratio = hs_coherence / (noise_coherence + 0.01)
    
    assert hs_coherence > noise_coherence, "Hyperspace should have higher coherence than noise"
    assert ratio > 5.0, f"Coherence ratio {ratio:.2f} too low for distinction"
    
    print("✓ Hyperspace waves distinguished from noise")
    print(f"  Hyperspace coherence: {hs_coherence:.4f}")
    print(f"  Noise coherence: {noise_coherence:.4f}")
    print(f"  Ratio: {ratio:.2f}x")
    print()
    return True


def test_complete_detection_system():
    """Test complete end-to-end detection."""
    print("Test 5: Complete Detection System")
    print("-" * 50)
    
    from hyperspace_wave_detector import HyperspaceWaveDetector
    
    detector = HyperspaceWaveDetector()
    detector.configure(
        carrier_freq=1e9,
        psi_freq=1e6,
        sampling_rate=1e10
    )
    
    # Run detection test
    results = detector.run_detection_test(
        duration=1e-6,
        noise_level=0.02,  # Low noise for reliable test
        n_psi_points=50
    )
    
    # Verify all required fields present
    required_fields = [
        'hyperspace_detected',
        'hyperspace_coherence',
        'em_coherence',
        'noise_coherence',
        'coherence_ratio_vs_em',
        'coherence_ratio_vs_noise',
    ]
    
    for field in required_fields:
        assert field in results, f"Missing field: {field}"
    
    # Check detection criteria
    assert results['hyperspace_coherence'] > 0.7, \
        f"Hyperspace coherence {results['hyperspace_coherence']:.4f} too low"
    
    assert results['coherence_ratio_vs_em'] > 3.0, \
        f"EM ratio {results['coherence_ratio_vs_em']:.2f} too low"
    
    assert results['coherence_ratio_vs_noise'] > 3.0, \
        f"Noise ratio {results['coherence_ratio_vs_noise']:.2f} too low"
    
    print("✓ Complete detection system works correctly")
    print(f"  Hyperspace detected: {results['hyperspace_detected']}")
    print(f"  Hyperspace coherence: {results['hyperspace_coherence']:.4f}")
    print(f"  EM ratio: {results['coherence_ratio_vs_em']:.2f}x")
    print(f"  Noise ratio: {results['coherence_ratio_vs_noise']:.2f}x")
    print()
    return True


def test_detection_with_noise():
    """Test detection under realistic noisy conditions."""
    print("Test 6: Detection Under Noise")
    print("-" * 50)
    
    from hyperspace_wave_detector import HyperspaceWaveDetector
    
    detector = HyperspaceWaveDetector()
    detector.configure(
        carrier_freq=1e9,
        psi_freq=1e6,
        sampling_rate=1e10
    )
    
    # Test with moderate noise
    results = detector.run_detection_test(
        duration=1e-6,
        noise_level=0.05,  # 5% noise (more realistic)
        n_psi_points=50
    )
    
    # Should still detect despite noise
    assert results['hyperspace_coherence'] > 0.4, \
        "Detection fails with moderate noise"
    
    print("✓ Detection works under noisy conditions")
    print(f"  Noise level: 5%")
    print(f"  Hyperspace coherence: {results['hyperspace_coherence']:.4f}")
    print(f"  Still detected: {results['hyperspace_detected']}")
    print()
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 70)
    print("HYPERSPACE WAVE DETECTOR TEST SUITE")
    print("=" * 70)
    print()
    
    tests = [
        test_transmitter_generation,
        test_receiver_signature_extraction,
        test_hyperspace_vs_em_distinction,
        test_hyperspace_vs_noise_distinction,
        test_complete_detection_system,
        test_detection_with_noise,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except AssertionError as e:
            print(f"✗ Test failed: {e}")
            print()
            failed += 1
        except Exception as e:
            print(f"✗ Test error: {e}")
            print()
            failed += 1
    
    print("=" * 70)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("=" * 70)
    print()
    
    if failed == 0:
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
