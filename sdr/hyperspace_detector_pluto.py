#!/usr/bin/env python3
"""
Hyperspace Wave Detector - ADALM-PLUTO SDR Implementation
Budget-friendly implementation using PlutoSDR ($150 USD)

This script provides a simpler, cheaper alternative to USRP hardware.
The PlutoSDR is ideal for this application: 325-3800 MHz, 20 MHz bandwidth.

Hardware Requirements:
- ADALM-PLUTO SDR ($150) - https://www.analog.com/en/design-center/evaluation-hardware-and-software/evaluation-boards-kits/adalm-pluto.html
- 2x SMA antennas (1 GHz) ($20 total)
- USB cable (included with Pluto)

Total cost: ~$170 (vs $1400+ for USRP)
"""

import adi
import numpy as np
import time
import sys

class HyperspaceDetectorPluto:
    """
    Hyperspace wave detector using ADALM-PLUTO SDR.
    
    Much cheaper and simpler than USRP, perfect for this experiment.
    """
    
    def __init__(self, carrier_freq=1e9, psi_freq=1e6, sample_rate=4e6, 
                 tx_gain=-10, rx_gain=60):
        """
        Initialize PlutoSDR detector.
        
        Parameters:
        -----------
        carrier_freq : float
            RF carrier frequency (default: 1 GHz)
        psi_freq : float
            Psi modulation frequency (default: 1 MHz)
        sample_rate : float
            Sample rate (max 20 MHz for Pluto, default: 4 MHz)
        tx_gain : float
            TX attenuation in dB (default: -10 dB)
        rx_gain : float
            RX gain in dB (default: 60 dB)
        """
        print("Initializing ADALM-PLUTO SDR...")
        
        # Create SDR interface
        try:
            self.sdr = adi.Pluto("ip:192.168.2.1")  # Default Pluto IP
        except:
            print("ERROR: Cannot connect to PlutoSDR at 192.168.2.1")
            print()
            print("Troubleshooting:")
            print("  1. Check USB connection")
            print("  2. Verify Pluto shows up as network device (192.168.2.1)")
            print("  3. Try: ping 192.168.2.1")
            print("  4. Update Pluto firmware if needed")
            raise
        
        # Configure sample rate
        self.sdr.sample_rate = int(sample_rate)
        
        # Configure TX
        self.sdr.tx_rf_bandwidth = int(sample_rate)
        self.sdr.tx_lo = int(carrier_freq)
        self.sdr.tx_hardwaregain_chan0 = tx_gain
        self.sdr.tx_cyclic_buffer = True  # Enable cyclic buffer for continuous TX
        
        # Configure RX
        self.sdr.rx_rf_bandwidth = int(sample_rate)
        self.sdr.rx_lo = int(carrier_freq)
        self.sdr.gain_control_mode_chan0 = "manual"
        self.sdr.rx_hardwaregain_chan0 = rx_gain
        self.sdr.rx_buffer_size = int(sample_rate * 0.1)  # 100ms buffer
        
        self.carrier_freq = carrier_freq
        self.psi_freq = psi_freq
        self.sample_rate = sample_rate
        
        print(f"✓ PlutoSDR initialized")
        print(f"  Sample rate: {sample_rate/1e6:.1f} MHz")
        print(f"  Carrier: {carrier_freq/1e9:.3f} GHz")
        print(f"  TX gain: {tx_gain} dB")
        print(f"  RX gain: {rx_gain} dB")
    
    def generate_hyperspace_signal(self, duration=1e-3):
        """
        Generate hyperspace wave signal with psi modulation.
        
        Parameters:
        -----------
        duration : float
            Signal duration in seconds
            
        Returns:
        --------
        signal : np.ndarray (complex)
            TX signal samples
        """
        n_samples = int(self.sample_rate * duration)
        t = np.arange(n_samples) / self.sample_rate
        
        # Create exponential envelope: exp(-2π·f_ψ·t)
        envelope = np.exp(-2 * np.pi * self.psi_freq * t)
        
        # Baseband signal (will be upconverted by Pluto to carrier_freq)
        # Just use the envelope as amplitude modulation
        signal = envelope * (1.0 + 0.0j)  # Convert to complex
        
        # Scale to Pluto's range [-1, 1] and convert to int16
        signal = signal * 0.5  # Reduce to avoid clipping
        
        return signal
    
    def generate_em_signal(self, duration=1e-3):
        """
        Generate EM control signal (no psi modulation).
        
        This is the control test - pure carrier with no envelope.
        """
        n_samples = int(self.sample_rate * duration)
        
        # Constant amplitude (no psi modulation)
        signal = np.ones(n_samples, dtype=complex) * 0.5
        
        return signal
    
    def transmit_signal(self, signal):
        """
        Transmit signal using PlutoSDR.
        
        Parameters:
        -----------
        signal : np.ndarray (complex)
            Signal to transmit
        """
        # Scale to int16 range for Pluto
        signal_scaled = (signal * 2**14).astype(np.int16)
        
        # Send to Pluto (cyclic buffer will repeat continuously)
        self.sdr.tx(signal_scaled)
    
    def receive_signal(self, duration=1e-3):
        """
        Receive signal samples.
        
        Parameters:
        -----------
        duration : float
            Capture duration in seconds
            
        Returns:
        --------
        samples : np.ndarray (complex)
            Received samples
        """
        n_samples = int(self.sample_rate * duration)
        
        # Receive samples
        samples = self.sdr.rx()
        
        return samples
    
    def stop_tx(self):
        """
        Stop transmission.
        """
        self.sdr.tx_destroy_buffer()
    
    def run_detection(self, test_duration=0.1):
        """
        Run complete hyperspace detection test.
        
        Parameters:
        -----------
        test_duration : float
            Test duration in seconds (default: 100ms)
            
        Returns:
        --------
        results : dict
            Detection results
        """
        print()
        print("=" * 70)
        print("RUNNING HYPERSPACE DETECTION TEST")
        print("=" * 70)
        print()
        
        # Test 1: Hyperspace signal
        print("Test 1: Hyperspace wave signal (with psi modulation)")
        print("  Generating signal...")
        hs_signal = self.generate_hyperspace_signal(duration=test_duration)
        
        print("  Transmitting...")
        self.transmit_signal(hs_signal)
        time.sleep(0.5)  # Allow TX to stabilize
        
        print("  Receiving...")
        hs_samples = self.receive_signal(duration=test_duration)
        
        # Extract envelope
        hs_envelope = np.abs(hs_samples)
        
        # Save to file
        hs_envelope.astype(np.float32).tofile('pluto_hyperspace_data.bin')
        print(f"  Saved {len(hs_envelope)} samples to pluto_hyperspace_data.bin")
        
        self.stop_tx()
        time.sleep(0.5)
        
        # Test 2: EM control signal
        print()
        print("Test 2: EM wave control (no psi modulation)")
        print("  Generating signal...")
        em_signal = self.generate_em_signal(duration=test_duration)
        
        print("  Transmitting...")
        self.transmit_signal(em_signal)
        time.sleep(0.5)
        
        print("  Receiving...")
        em_samples = self.receive_signal(duration=test_duration)
        
        # Extract envelope
        em_envelope = np.abs(em_samples)
        
        # Save to file
        em_envelope.astype(np.float32).tofile('pluto_em_data.bin')
        print(f"  Saved {len(em_envelope)} samples to pluto_em_data.bin")
        
        self.stop_tx()
        time.sleep(0.5)
        
        # Test 3: Noise (TX off)
        print()
        print("Test 3: Noise baseline (TX disabled)")
        print("  TX disabled, receiving noise...")
        
        noise_samples = self.receive_signal(duration=test_duration)
        noise_envelope = np.abs(noise_samples)
        
        noise_envelope.astype(np.float32).tofile('pluto_noise_data.bin')
        print(f"  Saved {len(noise_envelope)} samples to pluto_noise_data.bin")
        
        print()
        print("=" * 70)
        print("DATA COLLECTION COMPLETE")
        print("=" * 70)
        print()
        print("Next step: Analyze the data")
        print("  python analyze_sdr_data.py pluto_hyperspace_data.bin --samp-rate 4e6")
        print("  python analyze_sdr_data.py pluto_em_data.bin --samp-rate 4e6")
        print("  python analyze_sdr_data.py pluto_noise_data.bin --samp-rate 4e6")
        print()
        
        return {
            'hyperspace_file': 'pluto_hyperspace_data.bin',
            'em_file': 'pluto_em_data.bin',
            'noise_file': 'pluto_noise_data.bin'
        }
    
    def close(self):
        """
        Clean up and close SDR.
        """
        try:
            self.stop_tx()
        except:
            pass
        # Pluto context closes automatically


def main():
    """
    Main entry point for PlutoSDR hyperspace detector.
    """
    print("=" * 70)
    print("HYPERSPACE WAVE DETECTOR - ADALM-PLUTO SDR")
    print("=" * 70)
    print()
    print("This is a budget-friendly implementation using PlutoSDR ($150)")
    print()
    print("Hardware Setup:")
    print("  1. Connect PlutoSDR to computer via USB")
    print("  2. Attach TX antenna to TX port (1 GHz)")
    print("  3. Attach RX antenna to RX port (1 GHz)")
    print("  4. Keep antennas at least 30 cm apart")
    print("  5. Use shielded environment if possible")
    print()
    print("Software Requirements:")
    print("  pip install pyadi-iio numpy matplotlib scipy")
    print()
    
    try:
        # Create detector
        detector = HyperspaceDetectorPluto(
            carrier_freq=1e9,      # 1 GHz (within Pluto's 325-3800 MHz range)
            psi_freq=1e6,          # 1 MHz psi modulation
            sample_rate=4e6,       # 4 MHz (Pluto supports up to 20 MHz)
            tx_gain=-10,           # -10 dB attenuation (safe starting point)
            rx_gain=60             # 60 dB RX gain
        )
        
        # Run detection
        results = detector.run_detection(test_duration=0.1)  # 100ms test
        
        # Close
        detector.close()
        
        print("Test complete!")
        print()
        print("To analyze results, run:")
        print("  cd ../")
        print("  python sdr/analyze_sdr_data.py sdr/pluto_hyperspace_data.bin --samp-rate 4e6")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
