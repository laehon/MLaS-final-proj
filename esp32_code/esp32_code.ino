#include <Wire.h> 
#include <BluetoothSerial.h> 

BluetoothSerial SerialBT; 

const int MPU = 0x68; 
const float ACCEL_SCALE = 4096.0; // ±8g 
const float GYRO_SCALE = 131.0;   // ±250°/s 

void setup() { 
  SerialBT.begin("ESP32_Volleyball"); 
  Wire.begin(22, 21); 
  delay(100); 

  // Wake up 
  Wire.beginTransmission(MPU); 
  Wire.write(0x6B); 
  Wire.write(0); 
  Wire.endTransmission(true); 

  // ±8g 
  Wire.beginTransmission(MPU); 
  Wire.write(0x1C); 
  Wire.write(0x10); 
  Wire.endTransmission(true); 

  // ±500°/s 
  Wire.beginTransmission(MPU); 
  Wire.write(0x1B); 
  Wire.write(0x08); 
  Wire.endTransmission(true); 
} 

void loop() { 
  long t = millis(); 

  Wire.beginTransmission(MPU); 
  Wire.write(0x3B); 
  Wire.endTransmission(false); 
  Wire.requestFrom(MPU, 14, true); 

  int16_t axR = Wire.read() << 8 | Wire.read(); 
  int16_t ayR = Wire.read() << 8 | Wire.read(); 
  int16_t azR = Wire.read() << 8 | Wire.read(); 
  Wire.read(); Wire.read(); 
  int16_t gxR = Wire.read() << 8 | Wire.read(); 
  int16_t gyR = Wire.read() << 8 | Wire.read(); 
  int16_t gzR = Wire.read() << 8 | Wire.read(); 

  float ax = axR / ACCEL_SCALE; 
  float ay = ayR / ACCEL_SCALE; 
  float az = azR / ACCEL_SCALE; 
  float gx = gxR / GYRO_SCALE; 
  float gy = gyR / GYRO_SCALE; 
  float gz = gzR / GYRO_SCALE; 

  SerialBT.print(ax, 4); SerialBT.print(","); 
  SerialBT.print(ay, 4); SerialBT.print(","); 
  SerialBT.print(az, 4); SerialBT.print(","); 
  SerialBT.print(gx, 4); SerialBT.print(","); 
  SerialBT.print(gy, 4); SerialBT.print(","); 
  SerialBT.println(gz, 4); 

  while (millis() - t < 10); // 100Hz 
}