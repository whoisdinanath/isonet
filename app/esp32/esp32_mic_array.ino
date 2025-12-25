/*
 * ESP32 4-Channel I2S Microphone Array Firmware
 * 
 * Captures audio from 4 I2S MEMS microphones and streams
 * via UDP to the IsoNet real-time app.
 * 
 * Hardware:
 * - ESP32 DevKit V1
 * - 4x INMP441 or SPH0645 I2S MEMS microphones
 * - Square array configuration (7cm sides)
 * 
 * Wiring (example for INMP441):
 * - VDD: 3.3V
 * - GND: GND
 * - WS:  GPIO 25 (shared)
 * - SCK: GPIO 26 (shared)
 * - SD:  GPIO 22, 21, 19, 18 (one per mic)
 * - L/R: GND (left channel) or VDD (right channel)
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <driver/i2s.h>

// ============================================================
// CONFIGURATION
// ============================================================

// WiFi credentials
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// UDP target (IsoNet app)
const char* UDP_HOST = "192.168.1.100";  // PC running IsoNet
const uint16_t UDP_PORT = 8080;

// Audio settings
const int SAMPLE_RATE = 16000;
const int BITS_PER_SAMPLE = 16;
const int NUM_CHANNELS = 4;
const int CHUNK_MS = 500;  // Send every 500ms
const int CHUNK_SAMPLES = (SAMPLE_RATE * CHUNK_MS) / 1000;

// I2S pins (shared WS and SCK)
const int I2S_WS = 25;
const int I2S_SCK = 26;

// Data pins for each microphone
const int I2S_SD_PINS[4] = {22, 21, 19, 18};

// Packet magic number
const uint16_t PACKET_MAGIC = 0xAE32;

// ============================================================
// GLOBALS
// ============================================================

WiFiUDP udp;
uint16_t packetSeqNum = 0;

// Audio buffer: interleaved [ch0, ch1, ch2, ch3, ch0, ch1, ...]
int16_t audioBuffer[CHUNK_SAMPLES * NUM_CHANNELS];
int bufferIndex = 0;

// Packet header structure
struct __attribute__((packed)) PacketHeader {
    uint16_t magic;
    uint16_t seqNum;
    uint32_t timestampMs;
};

// ============================================================
// I2S CONFIGURATION
// ============================================================

// Note: ESP32's I2S peripheral supports up to 2 channels per port.
// For 4 channels, we need to use TDM mode or multiple I2S ports.
// This example uses a simplified approach with time-division reading.

void setupI2S() {
    // Configure I2S for first microphone pair (Port 0)
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 256,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };
    
    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SD_PINS[0]  // First mic
    };
    
    i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_NUM_0, &pin_config);
    
    Serial.println("I2S initialized");
}

// ============================================================
// WIFI SETUP
// ============================================================

void setupWiFi() {
    Serial.print("Connecting to WiFi");
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 30) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println();
        Serial.print("Connected! IP: ");
        Serial.println(WiFi.localIP());
    } else {
        Serial.println();
        Serial.println("WiFi connection failed!");
    }
}

// ============================================================
// AUDIO CAPTURE
// ============================================================

void captureAudio() {
    // Read from I2S
    // Note: This simplified version reads from one I2S port.
    // For true 4-channel, you would need TDM or multiple ports.
    
    size_t bytesRead = 0;
    int16_t samples[64];
    
    while (bufferIndex < CHUNK_SAMPLES) {
        i2s_read(I2S_NUM_0, samples, sizeof(samples), &bytesRead, portMAX_DELAY);
        
        int samplesRead = bytesRead / sizeof(int16_t) / 2;  // Stereo -> samples per channel
        
        for (int i = 0; i < samplesRead && bufferIndex < CHUNK_SAMPLES; i++) {
            // For demo: duplicate stereo to 4 channels
            // In real setup, you would read from all 4 mics
            int16_t left = samples[i * 2];
            int16_t right = samples[i * 2 + 1];
            
            // Interleave: [ch0, ch1, ch2, ch3]
            audioBuffer[bufferIndex * NUM_CHANNELS + 0] = left;
            audioBuffer[bufferIndex * NUM_CHANNELS + 1] = right;
            audioBuffer[bufferIndex * NUM_CHANNELS + 2] = left;  // Duplicate for demo
            audioBuffer[bufferIndex * NUM_CHANNELS + 3] = right; // Duplicate for demo
            
            bufferIndex++;
        }
    }
}

// ============================================================
// PACKET TRANSMISSION
// ============================================================

void sendPacket() {
    // Create packet
    const int headerSize = sizeof(PacketHeader);
    const int audioSize = CHUNK_SAMPLES * NUM_CHANNELS * sizeof(int16_t);
    const int packetSize = headerSize + audioSize;
    
    uint8_t* packet = (uint8_t*)malloc(packetSize);
    if (!packet) {
        Serial.println("Memory allocation failed!");
        return;
    }
    
    // Fill header
    PacketHeader* header = (PacketHeader*)packet;
    header->magic = PACKET_MAGIC;
    header->seqNum = packetSeqNum++;
    header->timestampMs = millis();
    
    // Copy audio data
    memcpy(packet + headerSize, audioBuffer, audioSize);
    
    // Send via UDP
    udp.beginPacket(UDP_HOST, UDP_PORT);
    udp.write(packet, packetSize);
    udp.endPacket();
    
    free(packet);
    
    // Reset buffer
    bufferIndex = 0;
}

// ============================================================
// MAIN
// ============================================================

void setup() {
    Serial.begin(115200);
    delay(1000);
    
    Serial.println();
    Serial.println("================================");
    Serial.println("ESP32 4-Channel Mic Array");
    Serial.println("================================");
    
    setupWiFi();
    setupI2S();
    
    udp.begin(UDP_PORT);
    
    Serial.println("Ready! Streaming to: " + String(UDP_HOST) + ":" + String(UDP_PORT));
}

void loop() {
    // Capture audio
    captureAudio();
    
    // Send when buffer is full
    if (bufferIndex >= CHUNK_SAMPLES) {
        sendPacket();
        
        // Debug: print every 10 packets
        if (packetSeqNum % 10 == 0) {
            Serial.printf("Sent packet %d (%d bytes)\n", 
                packetSeqNum - 1, 
                sizeof(PacketHeader) + CHUNK_SAMPLES * NUM_CHANNELS * sizeof(int16_t));
        }
    }
}
