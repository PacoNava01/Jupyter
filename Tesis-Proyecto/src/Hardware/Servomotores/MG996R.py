from adafruit_servokit import ServoKit
import time

def init_servos(num_servos=2, PWR=(405, 2500), act_range=180):

    kit = ServoKit(channels=16)

    for i in range(num_servos):
        kit.servo[i].set_pulse_width_range(PWR[0], PWR[1])
        kit.servo[i].actuation_range = act_range

    # asignación semántica
    servo_x = kit.servo[0]  # horizontal
    servo_y = kit.servo[1]  # vertical

    return servo_x, servo_y


def Servo2Pos(servo, angle):
    servo.angle = angle

servo_x,_ = init_servos()

if __name__ == "__main__":
    '''for i in range(0,181):
        Servo2Pos(servo_x,i)
        time.sleep(0.05)'''
    Servo2Pos(servo_x,0)
    time.sleep(2)
    Servo2Pos(servo_x,90)
    time.sleep(2)
    Servo2Pos(servo_x,180)
    time.sleep(2)
    Servo2Pos(servo_x,90)

    
    
    
    '''Servo2Pos(servo_x,90)
    time.sleep(2)
    Servo2Pos(servo_x,0)
    time.sleep(2)
    Servo2Pos(servo_x,90)
    time.sleep(2)
    Servo2Pos(servo_x,180)'''
