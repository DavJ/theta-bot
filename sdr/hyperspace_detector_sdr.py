#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperspace Wave Detector - GNU Radio Implementation
Requires: GNU Radio 3.10+, gr-osmosdr or gr-uhd, numpy, scipy

This script implements a complete SDR-based hyperspace wave detector
using GNU Radio flowgraph with USRP or compatible SDR hardware.
"""

from gnuradio import gr, blocks, analog, filter
from gnuradio import uhd  # For USRP SDR (or use osmosdr for other SDRs)
from gnuradio.filter import firdes
import numpy as np
import time
import sys

class HyperspaceDetectorSDR(gr.top_block):
    """
    GNU Radio flowgraph for hyperspace wave detection using SDR.
    
    Implements:
    - TX: Generates carrier with psi-modulated amplitude envelope
    - RX: Receives and demodulates signal to extract envelope
    - Saves received data for offline analysis
    """
    
    def __init__(self, samp_rate=10e6, carrier_freq=1e9, psi_freq=1e6, tx_gain=40, rx_gain=60):
        gr.top_block.__init__(self, "Hyperspace Detector SDR")
        
        self.samp_rate = samp_rate
        self.carrier_freq = carrier_freq
        self.psi_freq = psi_freq
        self.tx_gain = tx_gain
        self.rx_gain = rx_gain
        
        ##################################################
        # Transmitter Chain
        ##################################################
        
        # 1. Psi signal generator (1 MHz sine wave for exponential envelope)
        self.psi_source = analog.sig_source_f(
            samp_rate, 
            analog.GR_SIN_WAVE, 
            psi_freq, 
            1.0,  # amplitude
            0     # offset
        )
        
        # 2. Create exponential envelope: exp(-2π·f_ψ·t)
        # Use a time-varying multiplier
        self.time_source = analog.sig_source_f(
            samp_rate,
            analog.GR_SAW_WAVE,
            1.0,  # Ramp over 1 second
            1.0/samp_rate,  # Increment per sample
            0
        )
        
        # Scale for exponential decay
        self.exp_scale = blocks.multiply_const_ff(-2*np.pi*psi_freq)
        
        # Exponential function
        self.exp_func = blocks.transcendental('exp', 'float')
        
        # 3. RF carrier generator (will be upconverted by USRP)
        self.carrier_source = analog.sig_source_c(
            samp_rate,
            analog.GR_COS_WAVE,
            0,  # Baseband (USRP will upconvert to carrier_freq)
            1.0,
            0
        )
        
        # 4. Amplitude modulator (multiply carrier by envelope)
        self.am_modulator = blocks.multiply_cc(1)
        
        # Convert envelope to complex for multiplication
        self.float_to_complex = blocks.float_to_complex(1)
        
        # 5. USRP Sink (transmitter)
        self.usrp_sink = uhd.usrp_sink(
            ",".join(("", "")),
            uhd.stream_args(
                cpu_format="fc32",
                channels=list(range(1)),
            ),
        )
        self.usrp_sink.set_samp_rate(samp_rate)
        self.usrp_sink.set_center_freq(carrier_freq, 0)
        self.usrp_sink.set_gain(tx_gain, 0)
        self.usrp_sink.set_antenna('TX/RX', 0)
        self.usrp_sink.set_bandwidth(samp_rate, 0)
        
        ##################################################
        # Receiver Chain
        ##################################################
        
        # 6. USRP Source (receiver)
        self.usrp_source = uhd.usrp_source(
            ",".join(("", "")),
            uhd.stream_args(
                cpu_format="fc32",
                channels=list(range(1)),
            ),
        )
        self.usrp_source.set_samp_rate(samp_rate)
        self.usrp_source.set_center_freq(carrier_freq, 0)
        self.usrp_source.set_gain(rx_gain, 0)
        self.usrp_source.set_antenna('RX2', 0)
        self.usrp_source.set_bandwidth(samp_rate, 0)
        
        # 7. Demodulator (extract envelope - magnitude of complex signal)
        self.complex_to_mag = blocks.complex_to_mag(1)
        
        # 8. Low-pass filter (keep psi modulation, remove carrier artifacts)
        self.lpf_taps = firdes.low_pass(
            1.0,                    # gain
            samp_rate,              # sampling rate
            psi_freq * 5,           # cutoff freq (5x psi freq)
            psi_freq,               # transition width
            window=firdes.WIN_HAMMING,
            beta=6.76
        )
        self.lpf = filter.fir_filter_fff(1, self.lpf_taps)
        
        # 9. File sink for data analysis
        self.file_sink = blocks.file_sink(
            gr.sizeof_float*1, 
            'hyperspace_rx_data.bin', 
            False
        )
        self.file_sink.set_unbuffered(False)
        
        # 10. Optional: Vector sink for real-time monitoring
        self.vector_sink = blocks.vector_sink_f(1)
        
        ##################################################
        # Connections
        ##################################################
        
        # Transmitter flow:
        # time → scale → exp → envelope
        # carrier × envelope → USRP TX
        self.connect((self.time_source, 0), (self.exp_scale, 0))
        self.connect((self.exp_scale, 0), (self.exp_func, 0))
        self.connect((self.exp_func, 0), (self.float_to_complex, 0))
        self.connect((self.float_to_complex, 0), (self.am_modulator, 0))
        self.connect((self.carrier_source, 0), (self.am_modulator, 1))
        self.connect((self.am_modulator, 0), (self.usrp_sink, 0))
        
        # Receiver flow:
        # USRP RX → magnitude → LPF → file sink
        self.connect((self.usrp_source, 0), (self.complex_to_mag, 0))
        self.connect((self.complex_to_mag, 0), (self.lpf, 0))
        self.connect((self.lpf, 0), (self.file_sink, 0))
        self.connect((self.lpf, 0), (self.vector_sink, 0))


def main():
    """
    Run the SDR hyperspace detector.
    """
    print("=" * 70)
    print("HYPERSPACE WAVE DETECTOR - SDR IMPLEMENTATION")
    print("=" * 70)
    print()
    print("Hardware Requirements:")
    print("  - USRP B200/B210 or compatible SDR with TX/RX capability")
    print("  - 2 antennas tuned to 1 GHz (or use single antenna with switch)")
    print("  - Shielded environment or RF chamber recommended")
    print("  - Minimum 20 dB isolation between TX and RX antennas")
    print()
    print("Software Requirements:")
    print("  - GNU Radio 3.10+")
    print("  - UHD (USRP Hardware Driver)")
    print("  - Python 3.7+")
    print()
    
    # Check for command line arguments
    duration = 1.0  # seconds
    if len(sys.argv) > 1:
        try:
            duration = float(sys.argv[1])
        except:
            print("Usage: python hyperspace_detector_sdr.py [duration_seconds]")
            sys.exit(1)
    
    # Configuration
    config = {
        'samp_rate': 10e6,       # 10 MHz sampling rate
        'carrier_freq': 1e9,     # 1 GHz carrier frequency
        'psi_freq': 1e6,         # 1 MHz psi modulation
        'tx_gain': 40,           # TX gain in dB (adjust for your setup)
        'rx_gain': 60            # RX gain in dB (adjust for your setup)
    }
    
    print("Configuration:")
    print(f"  Sample Rate: {config['samp_rate']/1e6:.1f} MHz")
    print(f"  Carrier Frequency: {config['carrier_freq']/1e9:.3f} GHz")
    print(f"  Psi Modulation: {config['psi_freq']/1e6:.1f} MHz")
    print(f"  TX Gain: {config['tx_gain']} dB")
    print(f"  RX Gain: {config['rx_gain']} dB")
    print(f"  Duration: {duration} seconds")
    print()
    
    # Create flowgraph
    print("Initializing SDR hardware...")
    try:
        tb = HyperspaceDetectorSDR(**config)
    except Exception as e:
        print(f"ERROR: Failed to initialize SDR: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Check that USRP is connected via USB 3.0")
        print("  2. Run 'uhd_find_devices' to verify hardware")
        print("  3. Check that no other program is using the SDR")
        sys.exit(1)
    
    print("Starting transmission and reception...")
    print()
    
    # Start flowgraph
    tb.start()
    
    # Run for specified duration
    try:
        for i in range(int(duration)):
            time.sleep(1)
            print(f"  {i+1}/{int(duration)} seconds elapsed...")
    except KeyboardInterrupt:
        print()
        print("Interrupted by user")
    
    # Stop flowgraph
    print()
    print("Stopping SDR...")
    tb.stop()
    tb.wait()
    
    print()
    print("=" * 70)
    print("CAPTURE COMPLETE")
    print("=" * 70)
    print()
    print("Output file: hyperspace_rx_data.bin")
    print(f"File size: {len(open('hyperspace_rx_data.bin', 'rb').read())//4} samples")
    print()
    print("Next steps:")
    print("  1. Analyze the captured data:")
    print("     python analyze_sdr_data.py hyperspace_rx_data.bin")
    print()
    print("  2. Compare with EM wave control (set psi_freq=0 and re-run)")
    print()
    print("  3. Compare with noise (disconnect TX antenna and re-run)")
    print()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram terminated by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
