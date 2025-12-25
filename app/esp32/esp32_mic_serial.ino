/*
 * ESP32 4-Channel I2S Microphone Array - Serial Streaming
 * 
 * Captures audio from 4 I2S MEMS microphones and streams
 * via USB Serial to the IsoNet real-time app.
 * 
 * Advantages of Serial over WiFi:
 * - No network configuration required
 * - More reliable (no packet loss)
 * - Lower latency
 * - Simpler code
 * 
 * Hardware:
 * - ESP32 DevKit V1
 * - 4x INMP441 or SPH0645 I2S MEMS microphones
 * - Square array configuration (7cm sides)
 */

#include <Arduino.h>
#include <driver/i2s.h>

// ============================================================
// CONFIGURATION
// ============================================================

// Serial settings
#define SERIAL_BAUDRATE 921600  // High speed for audio streaming

// Audio settings
const int SAMPLE_RATE = 16000;
const int BITS_PER_SAMPLE = 16;
const int NUM_CHANNELS = 4;
const int CHUNK_MS = 100;  // Send every 100ms (smaller chunks for lower latency)
const int CHUNK_SAMPLES = (SAMPLE_RATE * CHUNK_MS) / 1000;

// I2S pins
const int I2S_WS = 25;    // Word Select (shared)
const int I2S_SCK = 26;   // Serial Clock (shared)
const int I2S_SD = 22;    // Serial Data (first mic pair)

// For true 4-channel, you need either:
// 1. TDM mode with 4 mics on same data line
// 2. Two I2S ports with 2 mics each
// This example uses stereo I2S duplicated to 4 channels for demo

// Packet magic number
const uint16_t PACKET_MAGIC = 0xAE32;

// ============================================================
// GLOBALS
// ============================================================

uint16_t packetSeqNum = 0;

// Audio buffer: interleaved [ch0, ch1, ch2, ch3, ...]
int16_t audioBuffer[CHUNK_SAMPLES * NUM_CHANNELS];

// DMA buffer for I2S
const int DMA_BUF_COUNT = 4;
const int DMA_BUF_LEN = 256;

// ============================================================
// PACKET STRUCTURE
// ============================================================

#pragma pack(push, 1)
struct PacketHeader {
    uint16_t magic;         // 0xAE32
    uint16_t seqNum;        // Packet sequence number
    uint32_t timestampMs;   // Milliseconds since boot
};
#pragma pack(pop)

// ============================================================
// I2S CONFIGURATION
// ============================================================

void setupI2S() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = DMA_BUF_COUNT,
        .dma_buf_len = DMA_BUF_LEN,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };
    
    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SD
    };
    
    esp_err_t err = i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
    if (err != ESP_OK) {
        Serial.printf("I2S driver install failed: %d\n", err);
        return;
    }
    
    err = i2s_set_pin(I2S_NUM_0, &pin_config);
    if (err != ESP_OK) {
        Serial.printf("I2S set pin failed: %d\n", err);
        return;
    }
    
    Serial.println("I2S initialized successfully");
}

// ============================================================
// AUDIO CAPTURE
// ============================================================

bool captureAudio() {
    size_t bytesRead = 0;
    int16_t stereoBuffer[DMA_BUF_LEN * 2];
    int bufferIndex = 0;
    
    while (bufferIndex < CHUNK_SAMPLES) {
        size_t toRead = min((int)(DMA_BUF_LEN * 2 * sizeof(int16_t)), 
                           (CHUNK_SAMPLES - bufferIndex) * 2 * sizeof(int16_t));
        
        esp_err_t err = i2s_read(I2S_NUM_0, stereoBuffer, toRead, &bytesRead, 100);
        if (err != ESP_OK) {
            Serial.printf("I2S read error: %d\n", err);
            return false;
        }
        
        int samplesRead = bytesRead / (2 * sizeof(int16_t));  // Stereo samples
        
        for (int i = 0; i < samplesRead && bufferIndex < CHUNK_SAMPLES; i++) {
            int16_t left = stereoBuffer[i * 2];
            int16_t right = stereoBuffer[i * 2 + 1];
            
            // Interleave to 4 channels: [ch0, ch1, ch2, ch3]
            // For demo: duplicate stereo to 4 channels
            // In real setup with 4 mics, read from all 4
            audioBuffer[bufferIndex * NUM_CHANNELS + 0] = left;
            audioBuffer[bufferIndex * NUM_CHANNELS + 1] = right;
            audioBuffer[bufferIndex * NUM_CHANNELS + 2] = left;   // Duplicate
            audioBuffer[bufferIndex * NUM_CHANNELS + 3] = right;  // Duplicate
            
            bufferIndex++;
        }
    }
    
    return true;
}

// ============================================================
// PACKET TRANSMISSION VIA SERIAL
// ============================================================

void sendPacketSerial() {
    // Create header
    PacketHeader header;
    header.magic = PACKET_MAGIC;
    header.seqNum = packetSeqNum++;
    header.timestampMs = millis();
    
    // Calculate sizes
    const int headerSize = sizeof(PacketHeader);
    const int audioSize = CHUNK_SAMPLES * NUM_CHANNELS * sizeof(int16_t);
    
    // Send header
    Serial.write((uint8_t*)&header, headerSize);
    
    // Send audio data
    Serial.write((uint8_t*)audioBuffer, audioSize);
    
    // Flush to ensure data is sent
    Serial.flush();
}

// ============================================================
// COMMAND HANDLING
// ============================================================

void handleSerialCommand() {
    if (Serial.available()) {
        char cmd = Serial.read();
        switch (cmd) {
            case 'S':  // Start streaming
                Serial.println("START");
                break;
            case 'P':  // Pause streaming
                Serial.println("PAUSE");
                break;
            case 'I':  // Info
                Serial.printf("INFO: SR=%d, CH=%d, CHUNK=%d\n", 
                             SAMPLE_RATE, NUM_CHANNELS, CHUNK_SAMPLES);
                break;
            case 'R':  // Reset
                packetSeqNum = 0;
                Serial.println("RESET");
                break;
        }
    }
}

// ============================================================
// MAIN
// ============================================================

void setup() {
    Serial.begin(SERIAL_BAUDRATE);
    
    // Wait for serial connection
    while (!Serial) {
        delay(10);
    }
    
    delay(1000);
    
    Serial.println();
    Serial.println("================================");
    Serial.println("ESP32 4-Channel Mic Array");
    Serial.println("Serial Streaming Mode");
    Serial.println("================================");
    Serial.printf("Baud Rate: %d\n", SERIAL_BAUDRATE);
    Serial.printf("Sample Rate: %d Hz\n", SAMPLE_RATE);
    Serial.printf("Channels: %d\n", NUM_CHANNELS);
    Serial.printf("Chunk: %d ms (%d samples)\n", CHUNK_MS, CHUNK_SAMPLES);
    
    int bytesPerChunk = sizeof(PacketHeader) + (CHUNK_SAMPLES * NUM_CHANNELS * 2);
    float chunksPerSecond = 1000.0 / CHUNK_MS;
    float bytesPerSecond = bytesPerChunk * chunksPerSecond;
    Serial.printf("Data Rate: %.1f KB/s\n", bytesPerSecond / 1024.0);
    Serial.println("================================");
    
    setupI2S();
    
    Serial.println("Ready! Streaming audio via Serial...");
    Serial.println();
}

void loop() {
    // Handle any incoming commands
    handleSerialCommand();
    
    // Capture audio
    if (captureAudio()) {
        // Send packet via Serial
        sendPacketSerial();
    }
    
    // Small delay to prevent watchdog issues
    yield();
}
