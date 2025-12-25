# ESP32 4-Channel Microphone Array Firmware

Firmware for ESP32 to capture 4-channel I2S audio and stream to IsoNet app.

## Hardware Requirements

- ESP32 DevKit V1 or similar
- 4x I2S MEMS microphones:
  - INMP441 (recommended)
  - SPH0645
  - ICS-43434
- 7cm x 7cm mounting board (square array)

## Wiring Diagram

```
                 ┌───────────────────┐
                 │     ESP32         │
                 │                   │
   ┌─────────────┤ GPIO 25 (WS)     │
   │ ┌───────────┤ GPIO 26 (SCK)    │
   │ │ ┌─────────┤ GPIO 22 (SD0)    │
   │ │ │ ┌───────┤ GPIO 21 (SD1)    │
   │ │ │ │ ┌─────┤ GPIO 19 (SD2)    │
   │ │ │ │ │ ┌───┤ GPIO 18 (SD3)    │
   │ │ │ │ │ │   │                   │
   │ │ │ │ │ │   │ 3.3V ──┬─┬─┬─┬──│
   │ │ │ │ │ │   │ GND  ──┼─┼─┼─┼──│
   │ │ │ │ │ │   └───────────────────┘
   │ │ │ │ │ │
   │ │ │ │ │ │    Mic Array (7cm square)
   │ │ │ │ │ │    ┌───────────────┐
   │ │ │ │ │ │    │ [0]       [1] │
   │ │ │ │ │ └────│──SD           │
   │ │ │ │ └──────│────────SD     │
   │ │ │ │        │               │
   │ │ │ │        │               │
   │ │ │ └────────│──SD           │
   │ │ └──────────│────────SD     │
   │ │            │ [2]       [3] │
   │ │            └───────────────┘
   │ │
   │ └─────── All mics share SCK
   └───────── All mics share WS

Mic connections:
- VDD → 3.3V
- GND → GND
- WS  → GPIO 25 (shared)
- SCK → GPIO 26 (shared)
- SD  → GPIO 22/21/19/18 (one per mic)
- L/R → GND (for left channel, all same)
```

## Microphone Positions

Square array with 7cm sides:

```
Position (meters):
  Mic 0: (-0.035, +0.035, 0)  # Top-left
  Mic 1: (+0.035, +0.035, 0)  # Top-right
  Mic 2: (-0.035, -0.035, 0)  # Bottom-left
  Mic 3: (+0.035, -0.035, 0)  # Bottom-right
```

## Configuration

Edit `esp32_mic_array.ino`:

```cpp
// WiFi credentials
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// UDP target (PC running IsoNet)
const char* UDP_HOST = "192.168.1.100";
const uint16_t UDP_PORT = 8080;
```

## Building

### Arduino IDE

1. Install ESP32 board support:

   - File → Preferences → Additional Boards Manager URLs:
   - Add: `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
   - Tools → Board → Boards Manager → Search "ESP32" → Install

2. Select board: Tools → Board → ESP32 Dev Module

3. Upload sketch

### PlatformIO

```bash
cd app/esp32
pio run -t upload
```

## Packet Format

UDP packets sent to IsoNet:

```
┌────────────────────────────────────────┐
│ Header (8 bytes)                       │
│ ├─ Magic: 0xAE32 (2 bytes, LE)        │
│ ├─ Sequence: uint16 (2 bytes, LE)     │
│ └─ Timestamp: uint32 ms (4 bytes, LE) │
├────────────────────────────────────────┤
│ Audio Data                             │
│ └─ int16 interleaved                  │
│    [ch0, ch1, ch2, ch3, ch0, ...]     │
└────────────────────────────────────────┘

Packet size for 500ms @ 16kHz:
  Header:  8 bytes
  Audio:   8000 samples × 4 ch × 2 bytes = 64000 bytes
  Total:   64008 bytes
```

## Testing

1. Flash firmware to ESP32
2. Open Serial Monitor (115200 baud)
3. Verify WiFi connection and IP
4. On PC, listen for packets:

```bash
# Linux/Mac
nc -u -l 8080 | xxd

# Python
python -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(('0.0.0.0', 8080))
while True:
    data, addr = s.recvfrom(65536)
    print(f'Received {len(data)} bytes from {addr}')
"
```

## Troubleshooting

### No audio / all zeros

- Check I2S wiring
- Verify L/R pin is connected to GND (for left channel)
- Try swapping WS/SCK connections

### WiFi not connecting

- Check SSID/password
- Ensure 2.4GHz network (ESP32 doesn't support 5GHz)

### Packet loss

- Reduce chunk size if network is slow
- Check WiFi signal strength
- Use wired Ethernet (ESP32-Ethernet-Kit)

### Audio artifacts

- Ensure proper power supply (3.3V, stable)
- Add decoupling capacitors near microphones
- Check ground connections
