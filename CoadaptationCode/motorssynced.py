from dynamixel_sdk import * 
import os
import threading
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt


'''
    For information on motor and it's look up tables:  https://emanual.robotis.com/docs/en/dxl/
'''

global timeToWriteList # global list to measure time it takes to write and reach positions, global because timers with locks/threading behave differently

class MotorsSynced:
    def __init__(self):

        # from robotis website
        if os.name == 'nt':
            import msvcrt
            def getch():
                return msvcrt.getch().decode()
        else:
            import sys, tty, termios
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            def getch():
                try:
                    tty.setraw(sys.stdin.fileno())
                    ch = sys.stdin.read(1)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                return ch
         

        # set motor bounds
        self.MIN_POS                        = float(1026) #change bounds based on design physical limits
        self.MAX_POS                        = float(3078) 
                        
        # set motor variables
        self.BAUDRATE                       = int(os.getenv("SNAKE_DXL_BAUDRATE", "2000000")) #57600 #2000000
        self.PROTOCOL_VERSION               = float(os.getenv("SNAKE_DXL_PROTOCOL_VERSION", "2.0")) # make sure motors are on this protocol version
        self.DXL_ID                         = list(reversed([1,2,3,4,5,6,7]))
        #[0,2,3,4,5,6]# IDs for motors, have these match to IDs set in dynamixel software
        self.ADDR_MX_TORQUE_ENABLE          = 64 # this ADDR value changes for different dynamixel models: https://emanual.robotis.com/docs/en/dxl/
        self.COMM_SUCCESS                   = 0 # variable for if message being sent to motors was successfully sent
        self.ADDR_GOAL_POSITION             = 116 # for writing position on table
        self.ADDR_PRESENT_POSITION          = 132 # for reading present position on table
        self.ADDR_PRESENT_VELOC             = 128 # for reading velocity
        self.ADDR_PRESENT_LOAD              = 126 # for reading variable similar to torque
        self.DXL_MOVING_STATUS_THRESHOLD    = 20 #11, higher threshold = faster mvmt but less accuracy in position Dynamixel moving status threshold, was 20
        

        self.LEN_GOAL_POS                   = 4 # data byte length
        self.LEN_PRES_POS                   = 4
        self.LEN_PRES_VELOC                 = 4
        self.LEN_PRES_LOAD                  = 2

        self.DEVICENAME                     = os.getenv("SNAKE_DXL_DEVICE", '/dev/ttyUSB1') # this changes with every device, in linux: '/dev/ttyUSB0'
        self.portHandler                    = PortHandler(self.DEVICENAME)
        self.packetHandler                  = PacketHandler(self.PROTOCOL_VERSION)

        # Initialize GroupSyncWrite instance
        self.groupSyncWrite = GroupSyncWrite(self.portHandler, self.packetHandler, self.ADDR_GOAL_POSITION, self.LEN_GOAL_POS)

        # Initialize GroupSyncRead instance for Present Position
        self.groupSyncRead = GroupSyncRead(self.portHandler, self.packetHandler, self.ADDR_PRESENT_POSITION, self.LEN_PRES_POS)

        # If want to read velocity
        #self.groupSyncReadVel = GroupSyncRead(self.portHandler, self.packetHandler, self.ADDR_PRESENT_VELOC, self.LEN_PRES_POS)

        self.groupSyncReadTor = GroupSyncRead(self.portHandler, self.packetHandler, self.ADDR_PRESENT_LOAD, self.LEN_PRES_LOAD) # for torque measurement


        # open the port
        if self.portHandler.openPort():
            print("Succeeded to open the port!")
        else:
            print("Failed to open the port!")
            quit()


        # set port baudrate
        if self.portHandler.setBaudRate(self.BAUDRATE):
            print("Succeeded to change the baudrate!")
        else:
            print("Failed to change the baudrate!")
            quit()
        
        # setting motor velocity
        for motorID in self.DXL_ID:
            dxlCommRes, dxlError = self.packetHandler.write4ByteTxRx(self.portHandler, motorID, 112, 500)
            self._log_motor_result(motorID, "set profile velocity", dxlCommRes, dxlError)
        time.sleep(.5)
        #enable torque
        self.enableTorque()

    def _log_motor_result(self, motorID, action, dxlCommRes, dxlError):
        if dxlCommRes != self.COMM_SUCCESS:
            print(f"Motor {motorID} {action} failed: {self.packetHandler.getTxRxResult(dxlCommRes)}")
            return False
        if dxlError != 0:
            print(f"Motor {motorID} {action} packet error: {self.packetHandler.getRxPacketError(dxlError)}")
            return False
        return True

    
    def setMotorSpeed(self):
         # setting motor velocity
        for motorID in self.DXL_ID:
            dxlCommRes, dxlError = self.packetHandler.write4ByteTxRx(self.portHandler, motorID, 112, 500) # change 500 to change motor velocity
            self._log_motor_result(motorID, "set profile velocity", dxlCommRes, dxlError)

    def enableTorque(self):
        # enable motor torques to be able to move motors     
        for motorID in self.DXL_ID:
            dxlCommRes, dxlError = self.packetHandler.write1ByteTxRx(self.portHandler, motorID, self.ADDR_MX_TORQUE_ENABLE, 1) # enable torque on motor
            if self._log_motor_result(motorID, "enable torque", dxlCommRes, dxlError):
                print("Dynamixel motor %i has been successfully connected" % motorID)

    
    def disableTorque(self): 
        # disable torques to lock motors    
        for motorID in self.DXL_ID:
            dxlCommRes, dxlError = self.packetHandler.write1ByteTxRx(self.portHandler, motorID, self.ADDR_MX_TORQUE_ENABLE, 0)
            self._log_motor_result(motorID, "disable torque", dxlCommRes, dxlError)
          
    def writePos(self,setPositionsTo):
        # write positions to motors
        try:
            if all((Pos >= self.MIN_POS and Pos <= self.MAX_POS) for Pos in setPositionsTo): # make sure all values are within bounds set 
                for motorID, setPos in zip(self.DXL_ID, setPositionsTo):
                    # add to parameter storage
                    setPosBytes = [DXL_LOBYTE(DXL_LOWORD(setPos)), DXL_HIBYTE(DXL_LOWORD(setPos)), DXL_LOBYTE(DXL_HIWORD(setPos)), DXL_HIBYTE(DXL_HIWORD(setPos))]     
                    addParamRes = self.groupSyncWrite.addParam(motorID, setPosBytes)
                    if addParamRes != True: # if couldn't add motor
                        print("Motor %i groupSyncwrite addparam failed" % motorID)
                        quit()

                dxlCommRes = self.groupSyncWrite.txPacket()# write goal positions
                self.groupSyncWrite.clearParam() # clears position storage
                
                if dxlCommRes != self.COMM_SUCCESS: # check if writing was a success
                        print("%s" % self.packetHandler.getTxRxResult(dxlCommRes))
        except:
            print('unable to send command position')
            self.groupSyncWrite.clearParam() # clears position storage
    
        
    def readPos(self):
        self.groupSyncRead.clearParam() # clear parameters from storage
        for motorID in self.DXL_ID: 
            addParamRes = self.groupSyncRead.addParam(motorID) # add parameters to be read
            if addParamRes != True:
                print("Motor %i groupSyncRead addparam failed" % motorID)
               

        motorPos = []


        # read present pos
        dxlCommRes = self.groupSyncRead.txRxPacket()
        if dxlCommRes != self.COMM_SUCCESS:
                print("groupSyncRead txRxPacket failed: %s" % self.packetHandler.getTxRxResult(dxlCommRes))
                self.groupSyncRead.clearParam()
                return []

        # see if groupsync data available then get data
        for motorID in self.DXL_ID:
            getDataRes = self.groupSyncRead.isAvailable(motorID, self.ADDR_PRESENT_POSITION, self.LEN_PRES_POS)
            #print(getDataRes) 
            if getDataRes != True:
                print("Motor %i groupSyncRead getdata failed" % motorID)
            else: # data is available
                motorPos.append(self.groupSyncRead.getData(motorID, self.ADDR_PRESENT_POSITION, self.LEN_PRES_POS))
       

        self.groupSyncRead.clearParam() # clear out data

        # normalize motor positions
        normalizedMotorPos = [2*(pos-self.MIN_POS)/(self.MAX_POS-self.MIN_POS)-1 for pos in motorPos]
        motorPos = normalizedMotorPos

        return motorPos
    
    def readVolt(self):
        self.groupSyncRead.clearParam() # clear parameters from storage
        for motorID in self.DXL_ID: 
            addParamRes = self.groupSyncRead.addParam(motorID) # add parameters to be read
            if addParamRes != True:
                print("Motor %i groupSyncRead addparam failed" % motorID)
               

        motorPos = []


        # read present pos
        dxlCommRes = self.groupSyncRead.txRxPacket()
        if dxlCommRes != self.COMM_SUCCESS:
                print("groupSyncRead txRxPacket failed: %s" % self.packetHandler.getTxRxResult(dxlCommRes))
                self.groupSyncRead.clearParam()
                return []

        ADR = 144
        LEN = 2
        # ADR = self.ADDR_PRESENT_POSITION
        # LEN = self.LEN_PRES_POS
        # see if groupsync data available then get data
        for motorID in self.DXL_ID:
            getDataRes = self.groupSyncRead.isAvailable(motorID, ADR, LEN)
            #print(getDataRes) 
            if getDataRes != True:
                print("Motor %i groupSyncRead getdata failed" % motorID)
            else: # data is available
                motorPos.append(self.groupSyncRead.getData(motorID, ADR, LEN))
       

        self.groupSyncRead.clearParam() # clear out data

        # # normalize motor positions
        # normalizedMotorPos = [2*(pos-self.MIN_POS)/(self.MAX_POS-self.MIN_POS)-1 for pos in motorPos]
        # motorPos = normalizedMotorPos

        return motorPos
    
    def readVeloc(self, lock):
        # FUNCTION NOT CURRENTLY IN USE

        # add groupsync reading parameters
        lock.acquire() # motor lock acquire
        for motorID in self.DXL_ID: 
            addParamRes = self.groupSyncReadVel.addParam(motorID) 
            if addParamRes != True:
                print("Motor %i groupSyncRead addparam failed" % motorID)
                quit()  

        motorVel = []

        # read present pos
        dxlCommRes = self.groupSyncReadVel.txRxPacket()
        if dxlCommRes != self.COMM_SUCCESS:
                print("groupSyncReadVel txRxPacket failed: %s" % self.packetHandler.getTxRxResult(dxlCommRes))
                self.groupSyncReadVel.clearParam()
                lock.release()
                return []

        # see if groupsync data available then get data
        for motorID in self.DXL_ID:
            getDataRes = self.groupSyncReadVel.isAvailable(motorID, self.ADDR_PRESENT_VELOC, self.LEN_PRES_POS)
            
            if getDataRes != True:
                print("Motor %i groupSyncRead getdata failed" % motorID)
                quit()
            else: # data is available
                motorVel.append(self.groupSyncReadVel.getData(motorID, self.ADDR_PRESENT_VELOC, self.LEN_PRES_POS))
       
        
        print('Current Velocity:', [float(i) for i in motorVel]) # print current position
        self.groupSyncReadVel.clearParam()
        lock.release() 

        return motorVel

    def readTorque(self, lock):
        # FUNCTION NOT CURRENTLY IN USE 

         # add groupsync reading parameters
        lock.acquire() # motor lock acquire
        for motorID in self.DXL_ID: 
            addParamRes = self.groupSyncReadTor.addParam(motorID) 
            if addParamRes != True:
                print("Motor %i groupSyncRead addparam failed" % motorID)
                quit()  

        motorTor = []

        # read present pos
        dxlCommRes = self.groupSyncReadTor.txRxPacket()
        if dxlCommRes != self.COMM_SUCCESS:
                print("groupSyncReadTor txRxPacket failed: %s" % self.packetHandler.getTxRxResult(dxlCommRes))
                self.groupSyncReadTor.clearParam()
                lock.release()
                return []

        # see if groupsync data available then get data
        for motorID in self.DXL_ID:
            getDataRes = self.groupSyncReadTor.isAvailable(motorID, self.ADDR_PRESENT_LOAD, self.LEN_PRES_LOAD)
            if getDataRes != True:
                print("Motor %i groupSyncRead getdata failed" % motorID)
                quit()
            else: # data is available
                motorTor.append(self.groupSyncReadTor.getData(motorID, self.ADDR_PRESENT_LOAD, self.LEN_PRES_LOAD))
       
   
        self.groupSyncReadTor.clearParam()
        lock.release() 

        return motorTor



    def endSequence(self):
        # disable torque
        self.disableTorque()

        # disable port
        self.portHandler.closePort()

    def rebootMotor(self, motor):
        # method to reboot motors
        
        dxlCommRes = self.packetHandler.reboot(self.portHandler, motor)
        if dxlCommRes != self.COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxlCommRes))
    

if __name__ == '__main__':
    testMotors = MotorsSynced()
    
    testMotors.setMotorSpeed()
    #print(testMotors.readPos())
    print(testMotors.readPos())
    testMotors.writePos([2025,2025,2025,2025,2025,2025,2025])
    time.sleep(5)
    for i in range(5):
        testMotors.writePos([1080,3050,1080,3050,1080,3050,1080])
        time.sleep(.3)
        # print(testMotors.readVolt())
        # time.sleep(.2)
        testMotors.writePos([3050,1080,3050,1080,3050,1080,3050])
        time.sleep(.3)
    # testMotors.writePos([1026])
    # time.sleep(1)
    # testMotors.writePos([3078])
    testMotors.disableTorque()
    
    print(testMotors.readPos())
    #testMotors.endSequence()
    

        
