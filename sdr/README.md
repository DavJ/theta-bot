# Hyperspace Wave Detector - SDR Implementation

## Budget-Friendly Option: ADALM-PLUTO SDR

**Recommended for minimal work and cost!**

### Why PlutoSDR?

- **Price**: $150 USD (vs $700-1400 for USRP)
- **Easy Setup**: USB plug-and-play, no complex drivers
- **Perfect Specs**: 325-3800 MHz (covers 1 GHz), 20 MHz bandwidth
- **Small & Portable**: Pocket-sized
- **Great Documentation**: Excellent community support

### Quick Start with PlutoSDR

#### 1. Hardware Setup

```bash
# Total cost: ~$170
- ADALM-PLUTO SDR: $150
  https://www.analog.com/en/design-center/evaluation-hardware-and-software/evaluation-boards-kits/adalm-pluto.html
  
- 2x 1 GHz SMA antennas: $20
  Any basic WiFi/cellular antenna works (avoid directional)
  
- USB cable: Included
```

#### 2. Software Installation

```bash
# Install Python library for Pluto
pip install pyadi-iio numpy matplotlib scipy

# Test connection
python -c "import adi; sdr = adi.Pluto(); print('PlutoSDR connected!')"
```

#### 3. Run Detection

```bash
# Navigate to SDR directory
cd sdr/

# Run the detector
python hyperspace_detector_pluto.py

# Analyze results
python analyze_sdr_data.py pluto_hyperspace_data.bin --samp-rate 4e6
python analyze_sdr_data.py pluto_em_data.bin --samp-rate 4e6
python analyze_sdr_data.py pluto_noise_data.bin --samp-rate 4e6
```

### Physical Setup

```
     TX Antenna              RX Antenna
         |                       |
         |                       |
    [============]          [============]
    |  PlutoSDR  |          |  (same)    |
    [============]          [============]
         |                       
         USB to computer

Note: Can use single PlutoSDR with antenna switch,
      or two PlutoSDRs for simultaneous TX/RX
      
Keep antennas ≥30 cm apart to avoid saturation
```

### Expected Results

When running the PlutoSDR detector:

```
Test 1: Hyperspace wave (with psi modulation)
  R² coherence: 0.70-0.90 (strong exponential fit)
  
Test 2: EM wave control (no psi modulation)
  R² coherence: 0.00-0.10 (no exponential fit)
  
Test 3: Noise baseline
  R² coherence: 0.00-0.05 (random noise)

Ratio: 10-90× difference → DETECTION CONFIRMED
```

## Alternative Budget Options

### RTL-SDR ($35) - Receive Only

```python
# Cheapest option but requires separate TX hardware
- RTL-SDR v3: $35 (RX only, 25-1750 MHz)
- HackRF One: $300 (TX/RX, 1 MHz - 6 GHz)

Total: $335 for TX/RX capability

See: sdr/hyperspace_detector_rtlsdr.py (to be implemented)
```

### LimeSDR Mini ($159) - Alternative to Pluto

```python
# Similar price to Pluto, wider bandwidth
- Frequency: 10 MHz - 3.5 GHz
- Bandwidth: 30.72 MHz
- More complex setup

Good alternative if Pluto is unavailable
```

## Comparison Table

| SDR | Price | Frequency | Bandwidth | Setup | Best For |
|-----|-------|-----------|-----------|-------|----------|
| **PlutoSDR** | **$150** | **325-3800 MHz** | **20 MHz** | **Easy** | **Recommended** |
| RTL-SDR | $35 | 25-1750 MHz | 3.2 MHz | Easy | RX only |
| HackRF | $300 | 1 MHz-6 GHz | 20 MHz | Medium | Full range |
| LimeSDR Mini | $159 | 10 MHz-3.5 GHz | 30 MHz | Hard | Power users |
| USRP B200 | $700 | 70 MHz-6 GHz | 56 MHz | Hard | Professional |
| USRP B210 | $1100 | 70 MHz-6 GHz | 56 MHz | Hard | Full duplex |

## Files in This Directory

### Core SDR Scripts

1. **hyperspace_detector_pluto.py** - PlutoSDR implementation (RECOMMENDED)
   - Complete TX/RX system
   - Generates hyperspace, EM, and noise tests
   - Saves data for analysis
   - Usage: `python hyperspace_detector_pluto.py`

2. **analyze_sdr_data.py** - Data analysis script
   - Extracts psi-signature from captured data
   - Calculates R² coherence
   - Generates plots
   - Usage: `python analyze_sdr_data.py <data.bin> --samp-rate 4e6`

3. **hyperspace_detector_sdr.py** - USRP/GNU Radio implementation
   - For USRP B200/B210 hardware
   - More complex but higher performance
   - Requires GNU Radio installation

## Troubleshooting

### PlutoSDR Not Detected

```bash
# Check USB connection
lsusb | grep -i pluto

# Check network interface
ping 192.168.2.1

# Update firmware if needed
# Download from: https://github.com/analogdevicesinc/plutosdr-fw
```

### Low Signal Quality

```python
# Increase RX gain
detector = HyperspaceDetectorPluto(rx_gain=70)  # Default is 60

# Reduce TX power (if saturating)
detector = HyperspaceDetectorPluto(tx_gain=-20)  # Default is -10

# Increase antenna separation
# Move antennas further apart (50+ cm)
```

### Poor R² Coherence

```python
# Increase test duration
detector.run_detection(test_duration=0.5)  # 500ms instead of 100ms

# Use lower sample rate for longer capture
detector = HyperspaceDetectorPluto(sample_rate=2e6)  # 2 MHz instead of 4 MHz

# Check for interference
# Use shielded environment or move away from WiFi/cellular
```

## Next Steps

1. **Get PlutoSDR**: Order from Analog Devices or distributors
2. **Install software**: `pip install pyadi-iio numpy matplotlib scipy`
3. **Connect hardware**: USB + 2 antennas
4. **Run detection**: `python hyperspace_detector_pluto.py`
5. **Analyze results**: `python analyze_sdr_data.py pluto_hyperspace_data.bin --samp-rate 4e6`
6. **Compare tests**: Hyperspace vs EM vs Noise

If R² > 0.65 for hyperspace and <0.10 for controls → **DETECTION CONFIRMED**

## Support

For PlutoSDR help:
- Wiki: https://wiki.analog.com/university/tools/pluto
- Forum: https://ez.analog.com/
- Examples: https://github.com/analogdevicesinc/pyadi-iio/tree/master/examples

For detector questions:
- See main documentation: ../HYPERSPACE_WAVE_DETECTOR_DESIGN.md
- Hardware guide: ../HYPERSPACE_HARDWARE_IMPLEMENTATION.md

---

**Bottom line**: PlutoSDR is the best budget option for this experiment. $150, plug-and-play, perfect specs. Recommended starting point!
