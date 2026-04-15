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
        self.ADDR_DRIVE_MODE                = 10
        self.ADDR_MX_TORQUE_ENABLE          = 64 # this ADDR value changes for different dynamixel models: https://emanual.robotis.com/docs/en/dxl/
        self.COMM_SUCCESS                   = 0 # variable for if message being sent to motors was successfully sent
        self.ADDR_PROFILE_ACCELERATION      = 108
        self.ADDR_PROFILE_VELOCITY          = 112
        self.ADDR_GOAL_POSITION             = 116 # for writing position on table
        self.ADDR_PRESENT_POSITION          = 132 # for reading present position on table
        self.ADDR_PRESENT_VELOC             = 128 # for reading velocity
        self.ADDR_PRESENT_LOAD              = 126 # for reading variable similar to torque
        self.ADDR_HARDWARE_ERROR_STATUS     = 70
        self.DXL_MOVING_STATUS_THRESHOLD    = 20 #11, higher threshold = faster mvmt but less accuracy in position Dynamixel moving status threshold, was 20
        self.DRIVE_MODE                     = int(os.getenv("SNAKE_DXL_DRIVE_MODE", "4"))
        self.PROFILE_ACCELERATION           = int(os.getenv("SNAKE_DXL_PROFILE_ACCELERATION", "25"))
        self.PROFILE_VELOCITY               = int(os.getenv("SNAKE_DXL_PROFILE_VELOCITY", "100"))
        reset_profile_acceleration = os.getenv("SNAKE_DXL_RESET_PROFILE_ACCELERATION")
        reset_profile_velocity = os.getenv("SNAKE_DXL_RESET_PROFILE_VELOCITY")
        self.RESET_PROFILE_ACCELERATION     = (
            int(reset_profile_acceleration)
            if reset_profile_acceleration is not None
            else None
        )
        self.RESET_PROFILE_VELOCITY         = (
            int(reset_profile_velocity)
            if reset_profile_velocity is not None
            else None
        )
        self.REBOOT_WAIT_SECONDS            = float(os.getenv("SNAKE_DXL_REBOOT_WAIT_S", "1.5"))
        self.RESET_TIMEOUT_SECONDS          = float(os.getenv("SNAKE_DXL_RESET_TIMEOUT_S", "10.0"))
        self.RESET_POLL_INTERVAL_SECONDS    = float(os.getenv("SNAKE_DXL_RESET_POLL_INTERVAL_S", "0.05"))
        self.RESET_POSITION_THRESHOLD       = int(os.getenv("SNAKE_DXL_RESET_THRESHOLD", "50"))
        self.RESET_MAX_STEP_COUNTS          = int(os.getenv("SNAKE_DXL_RESET_MAX_STEP_COUNTS", "80"))
        self.RESET_STAGE_TIMEOUT_SECONDS    = float(os.getenv("SNAKE_DXL_RESET_STAGE_TIMEOUT_S", "4.0"))
        self.RESET_STAGE_SETTLE_SECONDS     = float(os.getenv("SNAKE_DXL_RESET_STAGE_SETTLE_S", "0.2"))
        self.REBOOT_STATUS_RETRIES          = int(os.getenv("SNAKE_DXL_REBOOT_STATUS_RETRIES", "8"))
        self.REBOOT_STATUS_INTERVAL_SECONDS = float(os.getenv("SNAKE_DXL_REBOOT_STATUS_INTERVAL_S", "0.25"))
        self.last_good_motor_pos            = [0.0] * len(self.DXL_ID)
        self._recovery_in_progress          = False
        self._drive_mode_supported          = True

        self.LEN_GOAL_POS                   = 4 # data byte length
        self.LEN_PRES_POS                   = 4
        self.LEN_PRES_VELOC                 = 4
        self.LEN_PRES_LOAD                  = 2

        self.DEVICENAME                     = os.getenv("SNAKE_DXL_DEVICE", '/dev/ttyUSB2') # this changes with every device, in linux: '/dev/ttyUSB0'
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
        
        # Configure the motion profile before torque is enabled.
        self._configure_motion_profile()
        time.sleep(.5)
        #enable torque
        self.enableTorque()

    def _normalize_position(self, position):
        return 2 * (float(position) - self.MIN_POS) / (self.MAX_POS - self.MIN_POS) - 1

    def _cache_last_good_positions(self, raw_positions):
        self.last_good_motor_pos = [self._normalize_position(pos) for pos in raw_positions]

    def _format_hardware_error_status(self, hardware_error):
        if hardware_error == 0:
            return "clear"
        return ", ".join(self._decode_hardware_error_status(hardware_error))

    def _read_hardware_error_status_raw(self, motorID):
        return self.packetHandler.read1ByteTxRx(
            self.portHandler,
            motorID,
            self.ADDR_HARDWARE_ERROR_STATUS,
        )

    def reportHardwareErrorStatus(self, motorID, context=""):
        hardwareError, dxlCommRes, dxlError = self._read_hardware_error_status_raw(motorID)
        message_context = f" {context}" if context else ""

        if dxlCommRes != self.COMM_SUCCESS:
            print(
                f"Motor {motorID}{message_context} hardware status unavailable "
                f"(comm {dxlCommRes}): {self.packetHandler.getTxRxResult(dxlCommRes)}"
            )
            return None

        if dxlError != 0:
            print(
                f"Motor {motorID}{message_context} hardware status packet flags "
                f"0x{dxlError:02X}: {self.packetHandler.getRxPacketError(dxlError)}"
            )

        print(
            f"Motor {motorID}{message_context} hardware error status code: "
            f"0x{hardwareError:02X} ({self._format_hardware_error_status(hardwareError)})"
        )
        return hardwareError

    def reportHardwareErrorStatuses(self, motor_ids=None, context=""):
        motor_ids = self.DXL_ID if motor_ids is None else motor_ids
        status_map = {}
        for motorID in motor_ids:
            status_map[motorID] = self.reportHardwareErrorStatus(motorID, context=context)
        return status_map

    def _log_motor_result(self, motorID, action, dxlCommRes, dxlError, include_hardware_status=True):
        if dxlCommRes != self.COMM_SUCCESS:
            print(
                f"Motor {motorID} {action} failed (comm {dxlCommRes}): "
                f"{self.packetHandler.getTxRxResult(dxlCommRes)}"
            )
            if include_hardware_status:
                self.reportHardwareErrorStatus(motorID, context=f"after failed '{action}'")
            return False
        if dxlError != 0:
            print(
                f"Motor {motorID} {action} packet error 0x{dxlError:02X}: "
                f"{self.packetHandler.getRxPacketError(dxlError)}"
            )
            if include_hardware_status:
                self.reportHardwareErrorStatus(motorID, context=f"after failed '{action}'")
            return False
        return True

    def _coerce_write_result(self, result):
        if isinstance(result, tuple):
            if len(result) >= 2:
                return result[0], result[1]
            if len(result) == 1:
                return result[0], 0
        return result, 0

    def _reset_sync_handlers(self):
        self.groupSyncWrite = GroupSyncWrite(
            self.portHandler,
            self.packetHandler,
            self.ADDR_GOAL_POSITION,
            self.LEN_GOAL_POS,
        )
        self.groupSyncRead = GroupSyncRead(
            self.portHandler,
            self.packetHandler,
            self.ADDR_PRESENT_POSITION,
            self.LEN_PRES_POS,
        )
        self.groupSyncReadTor = GroupSyncRead(
            self.portHandler,
            self.packetHandler,
            self.ADDR_PRESENT_LOAD,
            self.LEN_PRES_LOAD,
        )

    def _configure_motion_profile(
        self,
        motor_ids=None,
        include_drive_mode=True,
        profile_acceleration=None,
        profile_velocity=None,
    ):
        motor_ids = self.DXL_ID if motor_ids is None else motor_ids
        profile_acceleration = (
            self.PROFILE_ACCELERATION
            if profile_acceleration is None
            else int(profile_acceleration)
        )
        profile_velocity = (
            self.PROFILE_VELOCITY
            if profile_velocity is None
            else int(profile_velocity)
        )
        success = True
        for motorID in motor_ids:
            if include_drive_mode and self._drive_mode_supported:
                dxlCommRes, dxlError = self.packetHandler.write1ByteTxRx(
                    self.portHandler,
                    motorID,
                    self.ADDR_DRIVE_MODE,
                    self.DRIVE_MODE,
                )
                if dxlCommRes == self.COMM_SUCCESS and dxlError == 0x07:
                    print(
                        f"Motor {motorID} drive mode address {self.ADDR_DRIVE_MODE} "
                        "is not supported on this model. Skipping future drive mode writes."
                    )
                    self._drive_mode_supported = False
                else:
                    success = self._log_motor_result(motorID, "set drive mode", dxlCommRes, dxlError) and success

            dxlCommRes, dxlError = self.packetHandler.write4ByteTxRx(
                self.portHandler,
                motorID,
                self.ADDR_PROFILE_ACCELERATION,
                profile_acceleration,
            )
            success = (
                self._log_motor_result(motorID, "set profile acceleration", dxlCommRes, dxlError)
                and success
            )

            dxlCommRes, dxlError = self.packetHandler.write4ByteTxRx(
                self.portHandler,
                motorID,
                self.ADDR_PROFILE_VELOCITY,
                profile_velocity,
            )
            success = (
                self._log_motor_result(motorID, "set profile velocity", dxlCommRes, dxlError)
                and success
            )

        return success

    def _build_reset_waypoints(self, start_positions, target_positions):
        if start_positions is None or len(start_positions) != len(target_positions):
            return [target_positions]

        max_step = max(1, int(self.RESET_MAX_STEP_COUNTS))
        deltas = [
            target_pos - start_pos
            for start_pos, target_pos in zip(start_positions, target_positions)
        ]
        max_delta = max(abs(delta) for delta in deltas)
        stages = max(1, int(np.ceil(max_delta / max_step)))
        waypoints = []

        for stage in range(1, stages + 1):
            fraction = stage / stages
            waypoint = [
                int(round(start_pos + (delta * fraction)))
                for start_pos, delta in zip(start_positions, deltas)
            ]
            if not waypoints or waypoint != waypoints[-1]:
                waypoints.append(waypoint)

        return waypoints

    def _uses_time_based_profile(self):
        return self._drive_mode_supported and bool(self.DRIVE_MODE & 0x04)

    def _reset_motion_profile_values(self):
        if self._uses_time_based_profile():
            default_acceleration = 300
            default_velocity = 1200
        else:
            default_acceleration = 10
            default_velocity = 40

        reset_acceleration = (
            default_acceleration
            if self.RESET_PROFILE_ACCELERATION is None
            else self.RESET_PROFILE_ACCELERATION
        )
        reset_velocity = (
            default_velocity
            if self.RESET_PROFILE_VELOCITY is None
            else self.RESET_PROFILE_VELOCITY
        )
        return reset_acceleration, reset_velocity

    def _restore_normal_motion_profile(self):
        reset_acceleration, reset_velocity = self._reset_motion_profile_values()
        if (
            reset_acceleration == self.PROFILE_ACCELERATION
            and reset_velocity == self.PROFILE_VELOCITY
        ):
            return True
        return self.setMotorSpeed()

    def _reopen_port(self):
        try:
            self.portHandler.closePort()
        except Exception:
            pass

        time.sleep(0.2)

        if not self.portHandler.openPort():
            print("Motor recovery failed: could not reopen the DYNAMIXEL port.")
            return False

        if not self.portHandler.setBaudRate(self.BAUDRATE):
            print("Motor recovery failed: could not restore the DYNAMIXEL baudrate.")
            return False

        self._reset_sync_handlers()
        return True

    def _decode_hardware_error_status(self, hardware_error):
        bit_labels = {
            0x01: "input voltage",
            0x04: "overheating",
            0x08: "encoder",
            0x10: "electrical shock",
            0x20: "overload",
        }
        decoded = [label for bit, label in bit_labels.items() if hardware_error & bit]
        return decoded if decoded else [f"unknown(0x{hardware_error:02X})"]

    def readHardwareErrorStatus(self, motorID):
        hardwareError, dxlCommRes, dxlError = self._read_hardware_error_status_raw(motorID)
        if dxlCommRes != self.COMM_SUCCESS:
            self._log_motor_result(
                motorID,
                "read hardware error status",
                dxlCommRes,
                dxlError,
                include_hardware_status=False,
            )
            return None
        if dxlError != 0:
            print(
                f"Motor {motorID} read hardware error status packet flags "
                f"0x{dxlError:02X}: {self.packetHandler.getRxPacketError(dxlError)}"
            )
        return hardwareError

    def _attempt_bus_recovery(self, context, motor_ids=None, force_reboot=False):
        if self._recovery_in_progress:
            print(f"Motor recovery already running; skipping nested recovery during {context}.")
            return False

        self._recovery_in_progress = True
        try:
            print(f"Attempting DYNAMIXEL recovery after: {context}")

            if not self._reopen_port():
                return False

            target_ids = list(self.DXL_ID if motor_ids is None else motor_ids)
            motors_to_reboot = []

            for motorID in target_ids:
                hardware_error = self.readHardwareErrorStatus(motorID)
                if hardware_error is None:
                    if force_reboot:
                        print(
                            f"Motor {motorID} hardware status unavailable after '{context}'. "
                            "Attempting reboot anyway."
                        )
                        motors_to_reboot.append(motorID)
                    continue
                if hardware_error != 0:
                    print(
                        f"Motor {motorID} latched hardware error status "
                        f"0x{hardware_error:02X} ({self._format_hardware_error_status(hardware_error)}). Rebooting."
                    )
                    motors_to_reboot.append(motorID)

            if not motors_to_reboot:
                print("No latched hardware errors found; re-applying motor settings.")
                self._configure_motion_profile(include_drive_mode=False)
                self.enableTorque()
                return True

            recovered_all = True
            for motorID in motors_to_reboot:
                recovered_all = self.rebootMotor(motorID) and recovered_all

            return recovered_all
        finally:
            self._recovery_in_progress = False

    def recoverFromFault(self, context, motor_ids=None, force_reboot=False):
        return self._attempt_bus_recovery(
            context,
            motor_ids=motor_ids,
            force_reboot=force_reboot,
        )

    
    def setMotorSpeed(self):
        # Re-apply the configured trapezoidal/time-based profile values.
        return self._configure_motion_profile()

    def _wait_for_clear_hardware_status(self, motorID):
        for _ in range(max(1, self.REBOOT_STATUS_RETRIES)):
            hardware_error = self.readHardwareErrorStatus(motorID)
            if hardware_error == 0:
                return True
            if hardware_error is not None:
                print(
                    f"Motor {motorID} still reports hardware error "
                    f"0x{hardware_error:02X} ({self._format_hardware_error_status(hardware_error)}) "
                    "after reboot."
                )
                return False
            time.sleep(self.REBOOT_STATUS_INTERVAL_SECONDS)

        print(f"Motor {motorID} did not respond with a clear hardware status after reboot.")
        return False

    def enableTorque(self):
        # enable motor torques to be able to move motors     
        success = True
        for motorID in self.DXL_ID:
            dxlCommRes, dxlError = self.packetHandler.write1ByteTxRx(self.portHandler, motorID, self.ADDR_MX_TORQUE_ENABLE, 1) # enable torque on motor
            if self._log_motor_result(motorID, "enable torque", dxlCommRes, dxlError):
                print("Dynamixel motor %i has been successfully connected" % motorID)
            else:
                success = False
        return success

    
    def disableTorque(self): 
        # disable torques to lock motors    
        success = True
        for motorID in self.DXL_ID:
            dxlCommRes, dxlError = self.packetHandler.write1ByteTxRx(self.portHandler, motorID, self.ADDR_MX_TORQUE_ENABLE, 0)
            success = self._log_motor_result(motorID, "disable torque", dxlCommRes, dxlError) and success
        return success

    def _read_positions_raw(self):
        self.groupSyncRead.clearParam()
        for motorID in self.DXL_ID:
            addParamRes = self.groupSyncRead.addParam(motorID)
            if addParamRes != True:
                print("Motor %i groupSyncRead addparam failed" % motorID)
                self.groupSyncRead.clearParam()
                return None

        motorPos = []
        unavailable_motors = []

        dxlCommRes = self.groupSyncRead.txRxPacket()
        if dxlCommRes != self.COMM_SUCCESS:
            print("groupSyncRead txRxPacket failed: %s" % self.packetHandler.getTxRxResult(dxlCommRes))
            self.groupSyncRead.clearParam()
            self._attempt_bus_recovery("position sync read failed", force_reboot=True)
            return None

        for motorID in self.DXL_ID:
            getDataRes = self.groupSyncRead.isAvailable(motorID, self.ADDR_PRESENT_POSITION, self.LEN_PRES_POS)
            if getDataRes != True:
                print("Motor %i groupSyncRead getdata failed" % motorID)
                unavailable_motors.append(motorID)
            else:
                motorPos.append(self.groupSyncRead.getData(motorID, self.ADDR_PRESENT_POSITION, self.LEN_PRES_POS))

        self.groupSyncRead.clearParam()

        if unavailable_motors:
            self._attempt_bus_recovery(
                f"missing position feedback from motors {unavailable_motors}",
                unavailable_motors,
                force_reboot=True,
            )
            return None

        return motorPos

    def waitForTargetPositions(self, target_positions, timeout_s=None):
        timeout_s = self.RESET_TIMEOUT_SECONDS if timeout_s is None else timeout_s
        deadline = time.time() + timeout_s
        last_positions = None
        last_errors = None

        while time.time() < deadline:
            motorPos = self._read_positions_raw()
            if motorPos is None:
                time.sleep(self.RESET_POLL_INTERVAL_SECONDS)
                continue

            self._cache_last_good_positions(motorPos)
            last_positions = motorPos
            last_errors = [
                abs(curr_pos - target_pos)
                for curr_pos, target_pos in zip(motorPos, target_positions)
            ]
            if all(
                error <= self.RESET_POSITION_THRESHOLD
                for error in last_errors
            ):
                return True

            time.sleep(self.RESET_POLL_INTERVAL_SECONDS)

        print(
            f"Timed out waiting for motors to reach reset position {target_positions} "
            f"within +/-{self.RESET_POSITION_THRESHOLD} counts after {timeout_s:.1f}s."
        )
        if last_positions is None:
            print("No valid motor position samples were received during reset wait.")
        else:
            print(f"Last motor positions: {last_positions}")
            print(f"Reset position errors: {last_errors}")
        self.reportHardwareErrorStatuses(context="after reset timeout")
        return False

    def resetMotorPositions(self, target_positions, disable_after_reset=False):
        if len(target_positions) != len(self.DXL_ID):
            print(
                f"Reset position count mismatch: expected {len(self.DXL_ID)} positions, "
                f"received {len(target_positions)}."
            )
            return False

        target_positions = [int(pos) for pos in target_positions]
        if not all(self.MIN_POS <= pos <= self.MAX_POS for pos in target_positions):
            print(f"Reset positions out of range: {target_positions}")
            return False

        reset_acceleration, reset_velocity = self._reset_motion_profile_values()
        print(
            f"Using reset motion profile: acceleration={reset_acceleration}, "
            f"velocity/time={reset_velocity}"
        )
        motion_profile_ok = self._configure_motion_profile(
            profile_acceleration=reset_acceleration,
            profile_velocity=reset_velocity,
        )
        torque_enabled = self.enableTorque()
        if not (motion_profile_ok and torque_enabled):
            status_map = self.reportHardwareErrorStatuses(context="during reset setup")
            latched_error_motors = [
                motorID
                for motorID, hardware_error in status_map.items()
                if hardware_error not in (None, 0)
            ]

            if latched_error_motors:
                print(
                    f"Attempting targeted reset recovery for motors with latched errors: "
                    f"{latched_error_motors}"
                )
                if self._attempt_bus_recovery("reset setup hardware fault", latched_error_motors):
                    print("Retrying reset after targeted motor recovery.")
                    motion_profile_ok = self._configure_motion_profile(
                        profile_acceleration=reset_acceleration,
                        profile_velocity=reset_velocity,
                    )
                    torque_enabled = self.enableTorque()

            if not (motion_profile_ok and torque_enabled):
                return False

        start_positions = self._read_positions_raw()
        waypoints = self._build_reset_waypoints(start_positions, target_positions)
        if start_positions is None:
            print("No current motor positions before reset; using one direct reset command.")
        else:
            print(
                f"Resetting motors from {start_positions} to {target_positions} "
                f"in {len(waypoints)} staged move(s)."
            )

        reached_target = True
        for waypoint_idx, waypoint in enumerate(waypoints):
            print(f"Reset stage {waypoint_idx + 1}/{len(waypoints)} target: {waypoint}")
            if not self.writePos(waypoint):
                self.reportHardwareErrorStatuses(context="after failed reset command")
                reached_target = False
                break

            stage_timeout_s = (
                self.RESET_TIMEOUT_SECONDS
                if waypoint_idx == len(waypoints) - 1
                else self.RESET_STAGE_TIMEOUT_SECONDS
            )
            if not self.waitForTargetPositions(waypoint, timeout_s=stage_timeout_s):
                reached_target = False
                break

            if waypoint_idx < len(waypoints) - 1:
                time.sleep(self.RESET_STAGE_SETTLE_SECONDS)

        if disable_after_reset:
            torque_disabled = self.disableTorque()
            normal_profile_ok = self._restore_normal_motion_profile()
            return reached_target and torque_disabled and normal_profile_ok

        normal_profile_ok = self._restore_normal_motion_profile()
        return reached_target and normal_profile_ok
          
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
                        self.groupSyncWrite.clearParam()
                        self._attempt_bus_recovery(
                            f"groupSyncWrite addParam failed for motor {motorID}",
                            [motorID],
                            force_reboot=True,
                        )
                        return False

                dxlCommRes = self.groupSyncWrite.txPacket()# write goal positions
                self.groupSyncWrite.clearParam() # clears position storage

                if dxlCommRes != self.COMM_SUCCESS: # check if writing was a success
                        print("%s" % self.packetHandler.getTxRxResult(dxlCommRes))
                        self.reportHardwareErrorStatuses(context="after failed goal position sync write")
                        self._attempt_bus_recovery("goal position sync write failed", force_reboot=True)
                        return False
                return True
            print(f"Requested motor positions out of bounds: {setPositionsTo}")
        except Exception as exc:
            print(f'unable to send command position: {exc}')
            self.groupSyncWrite.clearParam() # clears position storage
            self._attempt_bus_recovery("exception while writing goal positions", force_reboot=True)
        return False
    
        
    def readPos(self):
        motorPos = self._read_positions_raw()
        if motorPos is None:
            return self.last_good_motor_pos.copy()

        self._cache_last_good_positions(motorPos)
        return self.last_good_motor_pos.copy()
    
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
        reboot_result = self.packetHandler.reboot(self.portHandler, motor)
        dxlCommRes, dxlError = self._coerce_write_result(reboot_result)
        reboot_packet_ok = self._log_motor_result(motor, "reboot", dxlCommRes, dxlError)
        if not reboot_packet_ok:
            print(
                f"Motor {motor} reboot packet reported an error; "
                "waiting briefly to verify whether the reboot still completed."
            )

        time.sleep(self.REBOOT_WAIT_SECONDS)
        if not self._wait_for_clear_hardware_status(motor):
            return False

        if not self._configure_motion_profile([motor]):
            return False

        dxlCommRes, dxlError = self.packetHandler.write1ByteTxRx(
            self.portHandler,
            motor,
            self.ADDR_MX_TORQUE_ENABLE,
            1,
        )
        return self._log_motor_result(motor, "re-enable torque after reboot", dxlCommRes, dxlError)
    

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
    

        
