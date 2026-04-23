#include <SimpleFOC.h>
#include <Wire.h>

// Initialize sensor with custom configuration
MagneticSensorI2C sensor = MagneticSensorI2C(AS5048_I2C);

// Initialize motor and driver
BLDCMotor motor = BLDCMotor(11);
BLDCDriver3PWM driver = BLDCDriver3PWM(10, 5, 6, 8);  // A,B,C,EN

float target_angle = 0.3f;
Commander command = Commander(Serial);
void doTarget(char* cmd) { command.scalar(&target_angle, cmd); }
void doMotor(char* cmd)  { command.motor(&motor, cmd); }

void setup() {
  Serial.begin(115200);
  while(!Serial) _delay(10);

  SimpleFOCDebug::enable(&Serial);

  // Speed up I2C                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
  Wire.begin();
  Wire.setClock(400000);

  // Use AS5048_I2C template and override the chip address to 0x40
  MagneticSensorI2CConfig_s cfg = AS5048_I2C;
  cfg.chip_address = 0x40;
  sensor = MagneticSensorI2C(cfg);

  // Initialize sensor and link it to the motor
  sensor.init();
  motor.linkSensor(&sensor);

  // Driver initialization
  driver.voltage_power_supply = 11.1f; // Supply voltage 11.1V
  driver.voltage_limit = 6.0f;     // Limit driver maximum voltage output
  if(!driver.init()){
    Serial.println("Driver init failed!");
    while(1) {}
  }
  motor.linkDriver(&driver);

  // Control and modulation
  motor.foc_modulation = FOCModulationType::SpaceVectorPWM;
  motor.controller = MotionControlType::angle;
  motor.torque_controller = TorqueControlType::voltage; // No current sampling -> voltage torque

  // Filter/parameters
  motor.LPF_velocity.Tf = 0.02f;

  motor.P_angle.P = 7.0f;

  motor.PID_velocity.P = 0.5f;
  motor.PID_velocity.I = 2.0f;
  motor.PID_velocity.D = 0.0f;

  motor.voltage_limit = 4.0f;        // FOC operating voltage upper limit
  motor.velocity_limit = 5.0f;       // rad/s

  // Motor initialization
  motor.init();

  // Align sensor
  motor.voltage_sensor_align = 6.0f; // Provide enough torque to locate the zero point
  Serial.println("-----------------------------------");
  Serial.println("Starting FOC Alignment... Watch the motor!");

  motor.initFOC();

  // Add serial command listeners
  command.add('T', doTarget, "target angle (rad)");
  command.add('M', doMotor,  "motor tuning");

  // Configure monitoring: output target angle and actual angle
  motor.useMonitoring(Serial);
  motor.monitor_variables = _MON_TARGET | _MON_ANGLE;

  Serial.println("Motor ready. Send 'T1.57' to move.");
  Serial.println("-----------------------------------");
  _delay(1000);
}

void loop() {
  motor.loopFOC();
  motor.move(target_angle);

  // Print status in real time and accept serial input
  // motor.monitor();
  command.run();
}