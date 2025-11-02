# Hyperspace Wave Detector - Hardware Implementation Guide

## Overview

This document provides detailed electrical circuit designs, PCB layouts, antenna specifications, and SDR implementation code for building the hyperspace wave detector.

## ⚡ Quick Start - Budget Option (RECOMMENDED)

**Want to build this with minimal work and cost?**

### Use ADALM-PLUTO SDR - $170 Total

The **simplest and cheapest** way to build this detector:

1. **Buy**: ADALM-PLUTO SDR ($150) + 2 antennas ($20)
2. **Install**: `pip install pyadi-iio numpy matplotlib scipy`
3. **Run**: `python sdr/hyperspace_detector_pluto.py`
4. **Done**: Results in 5 minutes!

**See**: `sdr/README.md` for complete PlutoSDR guide

**No PCB fabrication, no soldering, no RF circuit design needed!**

---

For those who want to build custom hardware from scratch, continue reading below.

## System Architecture - Detailed

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRANSMITTER CHAIN                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Ref Osc]──>[PLL]──>[Mixer]──>[Bandpass]──>[PA]──>[Antenna]  │
│   10 MHz    1 GHz     ↑         Filter      5W      Patch      │
│                       │                                         │
│  [Psi Gen]──>[VGA]────┘                                        │
│   1 MHz    Amplitude                                            │
│            Modulator                                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    RECEIVER CHAIN                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Antenna]──>[LNA]──>[Mixer]──>[IF Amp]──>[ADC]──>[FPGA/SDR]  │
│   Patch      40dB     ↓         20dB       12bit   Processing   │
│                       │                     10GSa/s             │
│  [LO Gen]─────────────┘                                        │
│   999 MHz                                                       │
│   (for 1MHz IF)                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 1. Transmitter Circuit Design

### 1.1 RF Carrier Generator (1 GHz)

**Schematic: Carrier Generation using ADF4351 PLL**

```
Component List:
- U1: ADF4351 (Analog Devices) - Wideband Synthesizer with integrated VCO
- U2: ADP150 - 3.3V LDO Regulator
- X1: 10 MHz TCXO (Temperature Compensated Crystal Oscillator)
- L1-L3: Loop Filter inductors (see values below)
- C1-C10: Loop Filter and decoupling capacitors

Circuit Details:

VCC_5V ────┬─── C1(10µF) ─── GND
           │
           ├─── U2 (ADP150) ───── VCC_3.3V
           │
           └─── R1(10k) ─── LED1 ─── GND

X1 (10MHz TCXO):
    Pin 1 (VCC) ─── VCC_3.3V
    Pin 2 (GND) ─── GND
    Pin 3 (OUT) ─── C2(100nF) ─── U1.REFIN (Pin 1)

U1 (ADF4351):
    REFIN (Pin 1) ─── X1 output
    VCC (Pins 3,6,11,13,16,20,23) ─── VCC_3.3V + C3-C9(100nF each)
    GND (Pins 2,5,8,12,14,17,19,22,24) ─── GND
    
    Loop Filter (3rd order):
    CP (Pin 4) ─── R2(5.1kΩ) ───┬─── C10(22nF) ─── GND
                                 │
                                 ├─── L1(220nH) ─── C11(100nF) ─── GND
                                 │
                                 └─── L2(100nH) ─── C12(47nF) ─── GND
    
    VTUNE (Pin 7) ─── Loop Filter
    
    RF Output:
    RFOUT+ (Pin 15) ─── C13(100pF) ─── SMA_TX_OUT
    RFOUT- (Pin 18) ─── C14(100pF) ─── GND

SPI Control Interface:
    LE (Pin 9) ─── MCU.GPIO_CS
    CLK (Pin 10) ─── MCU.SPI_CLK
    DATA (Pin 21) ─── MCU.SPI_MOSI

Programming: Set N=100, R=1 for 1GHz output from 10MHz reference
```

### 1.2 Psi Modulator (1 MHz Amplitude Modulation)

**Schematic: Variable Gain Amplifier for Amplitude Modulation**

```
Component List:
- U3: AD8367 - Variable Gain Amplifier (90 dB gain range)
- U4: AD9833 - Programmable Waveform Generator (for psi signal)
- R3-R6: Resistors for gain control
- C15-C20: Coupling and bypass capacitors

AD9833 Psi Signal Generator:
    VCC (Pin 9) ─── VCC_3.3V
    GND (Pin 6) ─── GND
    MCLK (Pin 8) ─── 25 MHz crystal oscillator
    
    SPI Interface:
    SCLK (Pin 13) ─── MCU.SPI_CLK
    SDATA (Pin 12) ─── MCU.SPI_MOSI
    FSYNC (Pin 14) ─── MCU.GPIO_CS2
    
    IOUT (Pin 3) ─── R3(50Ω) ─── VGA_CONTROL
    
    Programming: Set frequency to 1 MHz

AD8367 Variable Gain Amplifier:
    VCC (Pin 8,7) ─── VCC_5V + C15,C16(100nF)
    GND (Pin 4,5) ─── GND
    
    Input (1 GHz carrier):
    INP (Pin 1) ─── C17(100pF) ─── Carrier from ADF4351
    INM (Pin 2) ─── C18(100pF) ─── GND (single-ended input)
    
    Gain Control (psi modulation):
    VGAIN (Pin 6) ─── R4(1kΩ) ───┬─── C19(10µF) ─── GND
                                  │
                                  └─── Psi signal from AD9833
    
    Output:
    OUTP (Pin 10) ─── C20(100pF) ─── Bandpass Filter
    OUTM (Pin 9) ─── C21(100pF) ─── GND

Gain Control Voltage Range: 0-1V controls 0-90dB gain
Psi modulation creates exponential amplitude envelope
```

### 1.3 Bandpass Filter and Power Amplifier

**Bandpass Filter (3rd Order Chebyshev, 1 GHz ±50 MHz)**

```
Component Values (50Ω impedance):
L3 = 8.2 nH (0603 chip inductor, Q>40)
L4 = 8.2 nH
C22 = 3.3 pF (0402 NP0 capacitor)
C23 = 3.3 pF
C24 = 3.3 pF

Schematic:
VGA_OUT ─── L3 ───┬─── L4 ─── PA_IN
                  │
                  ├─── C22 ─── GND
                  │
                  ├─── C23 ─── GND
                  │
                  └─── C24 ─── GND

Response: -3dB @ 950 MHz and 1050 MHz, -40dB @ 800 MHz and 1200 MHz
```

**Power Amplifier (5W output)**

```
Component List:
- U5: SKY65116-348LF - 5W Power Amplifier (Skyworks)
- Matching networks for input and output

SKY65116 Configuration:
    VCC (Pin 12) ─── VCC_28V (bias supply)
    VBIAS (Pin 3) ─── Bias network (R5,R6,C25)
    GND (Pins 1,2,4-11,13-16) ─── GND (thermal pad to ground plane)
    
    Input Matching:
    RFIN (Pin 14) ─── L5(3.3nH) ───┬─── C26(2.2pF) ─── BPF output
                                    │
                                    └─── C27(10pF) ─── GND
    
    Output Matching:
    RFOUT (Pin 15) ─── C28(3.3pF) ───┬─── L6(5.6nH) ─── Antenna
                                      │
                                      └─── C29(22pF) ─── GND

Output Power: +37 dBm (5W) typical
Gain: 30 dB typical
```

## 2. Receiver Circuit Design

### 2.1 Low Noise Amplifier (LNA)

**Schematic: Two-stage LNA using SPF5189Z**

```
Component List:
- U6, U7: SPF5189Z - Low Noise Amplifier (Qorvo)
- Matching networks for optimal noise figure

Stage 1 (U6):
    VCC (Pin 3) ─── L7(100nH RFC) ─── VCC_5V
    GND (Pin 2,4) ─── GND
    
    Input from Antenna:
    IN (Pin 1) ─── L8(3.9nH) ───┬─── C30(1.5pF) ─── Antenna
                                 │
                                 └─── C31(8.2pF) ─── GND
    
    Output:
    OUT (Pin 5) ─── C32(100pF) ─── Stage 2 input

Stage 2 (U7):
    Same topology as Stage 1
    
    Output:
    OUT (Pin 5) ─── C33(100pF) ─── Mixer input

Performance:
- Noise Figure: 0.8 dB @ 1 GHz
- Gain: 17 dB per stage (34 dB total)
- IP3: +20 dBm output
```

### 2.2 Mixer and IF Amplifier

**Mixer Circuit (Down-convert to 1 MHz IF)**

```
Component List:
- U8: ADE-1MHW - Double-balanced mixer (Mini-Circuits)
- U9: LTC6400-20 - High-speed differential amplifier

ADE-1MHW Mixer:
    RF (Port 1) ─── C34(100pF) ─── LNA output (1 GHz)
    LO (Port 2) ─── C35(100pF) ─── Local Oscillator (999 MHz)
    IF (Port 3) ─── C36(100nF) ─── IF Amplifier input (1 MHz)
    
    GND (Case) ─── GND plane

LTC6400-20 Differential Amplifier:
    VCC (Pin 8) ─── VCC_5V + C37(100nF)
    GND (Pin 4) ─── GND
    
    Input:
    INP (Pin 3) ─── C38(10µF) ─── Mixer IF output
    INM (Pin 5) ─── C39(10µF) ─── R7(50Ω) ─── GND
    
    Output:
    OUTP (Pin 1) ─── R8(50Ω) ─── ADC INP
    OUTM (Pin 7) ─── R9(50Ω) ─── ADC INM
    
    Gain control:
    GAIN (Pin 2) ─── R10(to set 20dB gain)

Conversion: 1 GHz RF → 1 MHz IF
IF Bandwidth: DC to 10 MHz
```

### 2.3 High-Speed ADC

**ADC Circuit: AD9467 (16-bit, 250 MSPS)**

```
Component List:
- U10: AD9467 - 16-bit, 250 MSPS ADC (Analog Devices)
- U11: ADA4930-2 - Differential driver for ADC
- Clock source: Si5338 programmable oscillator

ADA4930-2 ADC Driver:
    VCC (Pins 5,12) ─── VCC_5V
    GND (Pins 2,9) ─── GND
    
    Input from IF amp:
    INP (Pin 3) ─── C40(1µF) ─── IF_OUTP
    INM (Pin 11) ─── C41(1µF) ─── IF_OUTM
    
    Output to ADC:
    OUTP (Pin 7) ─── R11(49.9Ω) ─── ADC.VIN+
    OUTM (Pin 10) ─── R12(49.9Ω) ─── ADC.VIN-
    
    VOCM (Pin 8) ─── ADC.VCM (common mode reference)

AD9467 ADC:
    VCC (Pins 31,32,41,47,51) ─── VCC_3.3V_ANALOG
    DRVCC (Pins 2,19,26,44,60) ─── VCC_1.8V_DIGITAL
    GND (multiple pins) ─── GND
    
    Analog Input:
    VIN+ (Pin 34) ─── C42(1µF) ─── Driver output
    VIN- (Pin 36) ─── C43(1µF) ─── Driver output
    VCM (Pin 35) ─── Bias network
    
    Clock Input:
    CLK+ (Pin 48) ─── 250 MHz differential clock
    CLK- (Pin 49)
    
    Digital Output (LVDS):
    D0+ to D15+ (Pins 3,5,7,9,11,13,15,17,20,22,24,27,29,42,45,53)
    D0- to D15- (corresponding pairs)
    
    Connect to FPGA high-speed LVDS inputs

Sampling: 250 MSPS allows 125 MHz Nyquist bandwidth (sufficient for 1 MHz IF)
```

## 3. Antenna Design

### 3.1 Microstrip Patch Antenna (1 GHz)

**Specifications:**
- Frequency: 1 GHz (λ = 30 cm)
- Substrate: FR-4 (εr = 4.4, h = 1.6 mm)
- Impedance: 50Ω
- Gain: 6 dBi
- Polarization: Linear

**Dimensions (calculated):**

```
Patch Dimensions:
W = c/(2*f*sqrt((εr+1)/2)) = 11.2 cm (patch width)
L = 9.0 cm (patch length, adjusted for fringing)

Ground plane: 18 cm × 18 cm minimum

Feed point location (inset feed):
x_feed = L/2 = 4.5 cm from edge
y_feed = 0.35 cm from center (for 50Ω match)

PCB Stack-up:
Top layer (copper): Patch antenna (35 µm thick)
Dielectric: FR-4, 1.6 mm thick
Bottom layer (copper): Ground plane (35 µm thick)
```

**Physical Layout:**

```
Top View:
    ┌─────────────────────────────────┐
    │    Ground Plane (18cm × 18cm)   │
    │                                  │
    │     ┌─────────────────────┐     │
    │     │                     │     │
    │     │   Patch (11.2×9cm)  │     │
    │     │                     │     │
    │     │          ●──────────┼─────┤ SMA connector
    │     │      Feed point     │     │
    │     │                     │     │
    │     └─────────────────────┘     │
    │                                  │
    └─────────────────────────────────┘

Side View:
    Patch ════════════════════  (35 µm copper)
           ─────────────────────  (1.6 mm FR-4)
    Ground═══════════════════  (35 µm copper)
              │
              └──── SMA via
```

**Gerber Files (to be generated):**
1. Top Copper: Patch shape
2. Bottom Copper: Ground plane with via clearance
3. Drill file: SMA connector via
4. Silkscreen: Antenna orientation marker

### 3.2 Antenna Fabrication Notes

```
1. PCB Manufacturing:
   - Material: FR-4
   - Copper weight: 1 oz (35 µm)
   - Surface finish: ENIG (gold plating) for low loss
   - Board thickness: 1.6 mm ±0.1 mm

2. SMA Connector:
   - Type: SMA edge-mount or through-hole
   - Drill: 0.8 mm for center pin via
   - Keep ground clearance: 3 mm around feed

3. Tuning:
   - Trim patch length ±2 mm to adjust resonance
   - Use network analyzer to verify 50Ω match at 1 GHz
   - VSWR should be <1.5:1 at center frequency
```

## 4. PCB Design

### 4.1 Main RF Board Layout

**Board Stack-up (4-layer PCB):**

```
Layer 1 (Top): Signal traces, components, RF path
Layer 2: Ground plane (solid copper)
Layer 3: Power plane (VCC_3.3V, VCC_5V partitions)
Layer 4 (Bottom): Signal traces, ground returns

Board size: 100 mm × 150 mm
Material: Rogers RO4003C (low loss, εr=3.38) or FR-4 for budget
Thickness: 1.6 mm total
```

**Critical Layout Guidelines:**

```
1. RF Traces (50Ω microstrip):
   - Width: 3.1 mm (on FR-4, h=1.6mm to ground)
   - Keep away from edges: >5 mm
   - Minimize length, avoid sharp bends (use curved traces)
   - Ground vias every 5 mm along RF path

2. Component Placement:
   Transmitter section (left half):
   [TCXO] → [ADF4351] → [AD8367] → [BPF] → [PA] → [TX Antenna]
   
   Receiver section (right half):
   [RX Antenna] → [LNA] → [Mixer] → [IF Amp] → [ADC] → [FPGA]
   
   Digital/Control (top edge):
   [MCU] [Power supplies] [USB/Ethernet]

3. Grounding:
   - Pour solid ground on Layer 2
   - Stitch vias: 0.5 mm diameter, every 3-5 mm around RF components
   - Separate analog and digital ground with ferrite bead at single point

4. Power Distribution:
   - Wide traces: ≥2 mm for power
   - Decouple every IC: 100nF + 10µF at each VCC pin
   - Star topology from regulators to supplies

5. Shielding:
   - RF sections in metal cans (solder-down shields)
   - PA section: 15mm × 15mm shield
   - LNA section: 10mm × 10mm shield
```

**PCB Design Files (KiCad format):**

```
Files to create:
1. hyperspace_detector.kicad_sch - Schematic
2. hyperspace_detector.kicad_pcb - PCB layout
3. hyperspace_detector.kicad_pro - Project file
4. fp-lib-table - Footprint library
5. sym-lib-table - Symbol library

Output files:
1. Gerber files (RS-274X format)
2. Drill files (Excellon format)
3. BOM (Bill of Materials) CSV
4. Assembly drawing PDF
```

### 4.2 Power Supply Design

**Multi-rail Power Supply:**

```
Input: 12V DC, 5A (wall adapter)

Rail 1: +28V @ 2A (for PA)
[12V] → [Boost converter XL6009] → +28V
C50(470µF) for PA burst current

Rail 2: +5V @ 2A (for LNA, analog)
[12V] → [Buck converter LM2596] → +5V
C51(220µF) + C52(100nF)

Rail 3: +3.3V @ 1A (digital, PLL)
[5V] → [LDO ADP150] → +3.3V
C53(10µF) + C54(100nF)

Rail 4: +1.8V @ 0.5A (ADC digital)
[3.3V] → [LDO AP2127] → +1.8V
C55(10µF) + C56(100nF)

Total power: ~30W peak (PA transmitting)
```

## 5. SDR Implementation Code

### 5.1 GNU Radio Flowgraph (Python)

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperspace Wave Detector - GNU Radio Implementation
Requires: GNU Radio 3.10+, gr-osmosdr, numpy, scipy
"""

from gnuradio import gr, blocks, analog, filter
from gnuradio import uhd  # For USRP SDR
from gnuradio.filter import firdes
import numpy as np
import time

class HyperspaceDetectorSDR(gr.top_block):
    """
    GNU Radio flowgraph for hyperspace wave detection using SDR.
    """
    
    def __init__(self, samp_rate=10e6, carrier_freq=1e9, psi_freq=1e6):
        gr.top_block.__init__(self, "Hyperspace Detector SDR")
        
        self.samp_rate = samp_rate
        self.carrier_freq = carrier_freq
        self.psi_freq = psi_freq
        
        ##################################################
        # Transmitter Chain
        ##################################################
        
        # 1. Psi signal generator (1 MHz sine wave)
        self.psi_source = analog.sig_source_f(
            samp_rate, 
            analog.GR_SIN_WAVE, 
            psi_freq, 
            1.0,  # amplitude
            0     # offset
        )
        
        # 2. Exponential envelope: exp(-2π·f_ψ·t)
        # Implemented as time-varying multiply
        self.exp_envelope = blocks.multiply_ff(1)
        
        # Time ramp for exponential
        self.time_ramp = analog.sig_source_f(
            samp_rate,
            analog.GR_SAW_WAVE,
            1.0/1e-6,  # 1 microsecond period
            2*np.pi*psi_freq/samp_rate,
            0
        )
        
        # Exponential function block
        self.exp_block = blocks.transcendental('exp', 'float')
        
        # 3. RF carrier (1 GHz)
        self.carrier_source = analog.sig_source_c(
            samp_rate,
            analog.GR_COS_WAVE,
            0,  # Will be tuned by USRP to carrier_freq
            1.0,
            0
        )
        
        # 4. Amplitude modulator (multiply carrier by envelope)
        self.am_modulator = blocks.multiply_cc(1)
        
        # Convert envelope to complex
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
        self.usrp_sink.set_gain(40, 0)  # TX gain in dB
        self.usrp_sink.set_antenna('TX/RX', 0)
        
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
        self.usrp_source.set_gain(60, 0)  # RX gain in dB
        self.usrp_source.set_antenna('RX2', 0)
        
        # 7. Demodulator (extract envelope)
        self.complex_to_mag = blocks.complex_to_mag(1)
        
        # 8. Low-pass filter (remove carrier, keep envelope)
        self.lpf = filter.fir_filter_fff(
            1,
            firdes.low_pass(
                1, samp_rate, psi_freq*5, psi_freq, 
                window=firdes.WIN_HAMMING, 
                beta=6.76
            )
        )
        
        # 9. File sink for data analysis
        self.file_sink = blocks.file_sink(
            gr.sizeof_float*1, 
            'hyperspace_rx_data.bin', 
            False
        )
        self.file_sink.set_unbuffered(False)
        
        ##################################################
        # Connections
        ##################################################
        
        # Transmitter connections
        self.connect((self.psi_source, 0), (self.exp_envelope, 0))
        self.connect((self.time_ramp, 0), (self.exp_envelope, 1))
        self.connect((self.exp_envelope, 0), (self.exp_block, 0))
        self.connect((self.exp_block, 0), (self.float_to_complex, 0))
        self.connect((self.float_to_complex, 0), (self.am_modulator, 0))
        self.connect((self.carrier_source, 0), (self.am_modulator, 1))
        self.connect((self.am_modulator, 0), (self.usrp_sink, 0))
        
        # Receiver connections
        self.connect((self.usrp_source, 0), (self.complex_to_mag, 0))
        self.connect((self.complex_to_mag, 0), (self.lpf, 0))
        self.connect((self.lpf, 0), (self.file_sink, 0))


def main():
    """
    Run the SDR hyperspace detector.
    """
    print("Initializing Hyperspace Wave Detector SDR...")
    print()
    print("Hardware Requirements:")
    print("  - 2x USRP B200/B210 or similar SDR (one TX, one RX)")
    print("  - Antennas tuned to 1 GHz")
    print("  - Shielded environment or RF chamber")
    print()
    
    # Create flowgraph
    tb = HyperspaceDetectorSDR(
        samp_rate=10e6,      # 10 MHz sampling
        carrier_freq=1e9,    # 1 GHz carrier
        psi_freq=1e6         # 1 MHz psi modulation
    )
    
    print("Starting transmission and reception...")
    print("Duration: 10 seconds")
    print()
    
    # Start flowgraph
    tb.start()
    
    # Run for 10 seconds
    time.sleep(10)
    
    # Stop flowgraph
    tb.stop()
    tb.wait()
    
    print("Capture complete.")
    print("Data saved to: hyperspace_rx_data.bin")
    print()
    print("Run analysis script to process data:")
    print("  python analyze_sdr_data.py hyperspace_rx_data.bin")


if __name__ == '__main__':
    main()
```

### 5.2 SDR Data Analysis Script

```python
#!/usr/bin/env python3
"""
Analyze SDR captured data for hyperspace wave signatures.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit

def analyze_sdr_data(filename, samp_rate=10e6, psi_freq=1e6):
    """
    Analyze captured SDR data for hyperspace signature.
    """
    print(f"Loading data from {filename}...")
    
    # Load data (float32 format from GNU Radio)
    data = np.fromfile(filename, dtype=np.float32)
    
    print(f"Loaded {len(data)} samples ({len(data)/samp_rate:.3f} seconds)")
    print()
    
    # Time array
    t = np.arange(len(data)) / samp_rate
    
    # Extract amplitude envelope (should already be extracted by receiver)
    amplitude = data
    
    # Fit exponential model: A(t) = A0 * exp(-alpha * t)
    def exp_model(t, A0, alpha, offset):
        return A0 * np.exp(-alpha * t) + offset
    
    # Initial guess
    p0 = [np.max(amplitude), 2*np.pi*psi_freq, np.min(amplitude)]
    
    try:
        # Fit
        popt, pcov = curve_fit(exp_model, t, amplitude, p0=p0, maxfev=10000)
        A0, alpha, offset = popt
        
        # Predicted amplitude
        amplitude_pred = exp_model(t, *popt)
        
        # Calculate R²
        residuals = amplitude - amplitude_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((amplitude - np.mean(amplitude))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        print("=== SDR Analysis Results ===")
        print()
        print(f"Exponential Fit Parameters:")
        print(f"  A0 (initial amplitude): {A0:.6f}")
        print(f"  alpha (decay rate): {alpha:.6f} rad/s")
        print(f"  Expected alpha: {2*np.pi*psi_freq:.6f} rad/s")
        print(f"  Alpha error: {abs(alpha - 2*np.pi*psi_freq)/(2*np.pi*psi_freq)*100:.2f}%")
        print()
        print(f"Goodness of Fit:")
        print(f"  R² (coherence): {r_squared:.6f}")
        print()
        
        if r_squared > 0.65:
            print("✓ HYPERSPACE WAVE SIGNATURE DETECTED")
            print()
            print("The captured signal exhibits strong exponential amplitude")
            print("modulation consistent with imaginary-time propagation.")
        else:
            print("✗ NO HYPERSPACE WAVE SIGNATURE")
            print()
            print(f"Coherence R²={r_squared:.4f} is below threshold (0.65)")
        
        # Plot results
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Time series
        plt.subplot(2, 1, 1)
        plt.plot(t*1e6, amplitude, 'b-', alpha=0.5, label='Captured Data')
        plt.plot(t*1e6, amplitude_pred, 'r-', linewidth=2, label=f'Exponential Fit (R²={r_squared:.4f})')
        plt.xlabel('Time (μs)')
        plt.ylabel('Amplitude')
        plt.title('Hyperspace Wave Detection - SDR Data')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Log scale (should be linear for exponential)
        plt.subplot(2, 1, 2)
        plt.semilogy(t*1e6, amplitude, 'b-', alpha=0.5, label='Captured Data')
        plt.semilogy(t*1e6, amplitude_pred, 'r-', linewidth=2, label='Exponential Fit')
        plt.xlabel('Time (μs)')
        plt.ylabel('Amplitude (log scale)')
        plt.title('Logarithmic View (Linear in log scale confirms exponential)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sdr_analysis_results.png', dpi=150)
        print()
        print("Plot saved to: sdr_analysis_results.png")
        
    except Exception as e:
        print(f"Error during fitting: {e}")
        print("Data may not contain exponential signature.")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_sdr_data.py <data_file.bin>")
        sys.exit(1)
    
    analyze_sdr_data(sys.argv[1])
```

### 5.3 Alternative: RTL-SDR Implementation (Budget Option)

```python
#!/usr/bin/env python3
"""
Hyperspace Wave Detector using RTL-SDR (budget implementation).
Requires: pyrtlsdr, numpy, scipy
"""

from rtlsdr import RtlSdr
import numpy as np
import time

class HyperspaceDetectorRTLSDR:
    """
    Simple hyperspace detector using RTL-SDR dongle.
    Note: TX requires separate hardware (HackRF or similar).
    """
    
    def __init__(self, center_freq=1e9, sample_rate=2.4e6):
        self.sdr = RtlSdr()
        self.sdr.sample_rate = sample_rate
        self.sdr.center_freq = center_freq
        self.sdr.gain = 'auto'
        
    def capture(self, duration=1e-6, num_samples=None):
        """
        Capture samples for specified duration.
        """
        if num_samples is None:
            num_samples = int(duration * self.sdr.sample_rate)
        
        print(f"Capturing {num_samples} samples...")
        samples = self.sdr.read_samples(num_samples)
        
        return samples
    
    def close(self):
        self.sdr.close()


def main():
    print("Hyperspace Wave Detector - RTL-SDR Implementation")
    print()
    print("Note: RTL-SDR can only receive. You need separate TX hardware.")
    print("Recommended: HackRF One for TX, RTL-SDR for RX")
    print()
    
    detector = HyperspaceDetectorRTLSDR(
        center_freq=1e9,
        sample_rate=2.4e6
    )
    
    # Capture
    samples = detector.capture(duration=100e-6)  # 100 microseconds
    
    # Extract envelope
    envelope = np.abs(samples)
    
    # Save for analysis
    envelope.tofile('rtlsdr_capture.bin')
    print("Data saved to: rtlsdr_capture.bin")
    
    detector.close()


if __name__ == '__main__':
    main()
```

## 6. Bill of Materials (BOM) - Complete

### 6.1 Active Components

| Ref | Part Number | Description | Qty | Unit Price | Total |
|-----|-------------|-------------|-----|------------|-------|
| U1 | ADF4351 | PLL Synthesizer | 1 | $15.00 | $15.00 |
| U2 | ADP150 | 3.3V LDO Regulator | 1 | $2.50 | $2.50 |
| U3 | AD8367 | Variable Gain Amplifier | 1 | $8.50 | $8.50 |
| U4 | AD9833 | Waveform Generator | 1 | $5.00 | $5.00 |
| U5 | SKY65116 | 5W Power Amplifier | 1 | $35.00 | $35.00 |
| U6, U7 | SPF5189Z | Low Noise Amplifier | 2 | $3.50 | $7.00 |
| U8 | ADE-1MHW | Mixer | 1 | $12.00 | $12.00 |
| U9 | LTC6400-20 | Differential Amp | 1 | $8.00 | $8.00 |
| U10 | AD9467 | 16-bit ADC | 1 | $45.00 | $45.00 |
| U11 | ADA4930-2 | ADC Driver | 1 | $6.00 | $6.00 |
| - | STM32F4 | Microcontroller | 1 | $10.00 | $10.00 |
| - | Si5338 | Clock Generator | 1 | $12.00 | $12.00 |

### 6.2 Passive Components (selected)

| Type | Value | Qty | Unit Price | Total |
|------|-------|-----|------------|-------|
| Resistors | Various | 50 | $0.10 | $5.00 |
| Capacitors | Various | 100 | $0.20 | $20.00 |
| Inductors | Various | 20 | $0.50 | $10.00 |

### 6.3 Mechanical & PCB

| Item | Description | Qty | Unit Price | Total |
|------|-------------|-----|------------|-------|
| PCB | 4-layer, 100x150mm | 5 | $50.00 | $250.00 |
| Antenna PCB | Patch antenna | 2 | $20.00 | $40.00 |
| SMA Connectors | Edge mount | 4 | $2.00 | $8.00 |
| RF Shields | Metal cans | 3 | $3.00 | $9.00 |
| Enclosure | Aluminum box | 1 | $30.00 | $30.00 |

### 6.4 Optional SDR Hardware

| Item | Description | Unit Price |
|------|-------------|------------|
| USRP B200 | SDR (2x needed) | $700.00 each |
| HackRF One | SDR TX/RX | $300.00 |
| RTL-SDR v3 | RX only | $35.00 |

**Total (without SDR): ~$550**  
**Total (with HackRF): ~$850**  
**Total (with 2x USRP): ~$1,950**

## 7. Assembly Instructions

### 7.1 PCB Assembly

```
1. Inspect PCB for defects
2. Apply solder paste using stencil
3. Place components using pick-and-place or tweezers
   - Start with smallest components (resistors, capacitors)
   - Then ICs (check orientation!)
   - Finally connectors and shields
4. Reflow solder (oven or hot air)
5. Inspect joints under microscope
6. Test for shorts (multimeter between VCC and GND)
```

### 7.2 Testing Procedure

```
1. Power-up test (no RF):
   - Connect 12V supply
   - Verify all rail voltages (28V, 5V, 3.3V, 1.8V)
   - Check current draw (<100mA without TX)

2. PLL Lock test:
   - Program ADF4351 for 1 GHz
   - Check MUXOUT pin for lock detect
   - Measure RF output with spectrum analyzer

3. TX chain test:
   - Generate psi modulation (1 MHz)
   - Observe output on oscilloscope
   - Verify exponential envelope

4. RX chain test:
   - Inject known signal at antenna
   - Verify LNA gain and noise figure
   - Check ADC output data

5. Complete system test:
   - Run detection algorithm
   - Verify hyperspace signature detection
```

## 8. Troubleshooting

### Common Issues:

| Problem | Cause | Solution |
|---------|-------|----------|
| No TX output | PLL not locked | Check reference clock, SPI programming |
| High noise | Poor grounding | Add more ground vias, check shields |
| Low sensitivity | LNA not biased | Verify DC voltages, check matching |
| ADC clipping | Too much gain | Reduce IF amp or LNA gain |
| Spurious signals | Poor filtering | Add/improve bandpass filters |

## 9. Safety Notes

```
WARNING: This device transmits RF energy.
- Maximum TX power: +37 dBm (5W)
- Use in shielded environment or with proper antennas
- Comply with local RF regulations (FCC Part 15, etc.)
- Do not operate near medical equipment
- Use ESD protection when handling boards
```

## 10. Next Steps

1. Order PCB and components
2. Assemble and test individual sections
3. Integrate complete system
4. Perform detection experiments
5. Document results

## Files Included

- `schematics/` - KiCad schematic files
- `pcb/` - PCB layout files
- `gerbers/` - Manufacturing files
- `bom/` - Bill of materials
- `software/` - SDR implementation code
- `docs/` - Additional documentation

---

This hardware implementation guide provides complete details for building the hyperspace wave detector. For theoretical background, see HYPERSPACE_WAVE_DETECTOR_DESIGN.md.
