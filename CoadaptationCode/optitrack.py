
# coding: utf-8
"""Command-line NatNet client application for testing.

Copyright (c) 2017, Matthew Edwards.  This file is subject to the 3-clause BSD
license, as found in the LICENSE file in the top-level directory of this
distribution and at https://github.com/mje-nz/python_natnet/blob/master/LICENSE.
No part of python_natnet, including this file, may be copied, modified,
propagated, or distributed except according to the terms contained in the
LICENSE file.
"""

# NatNet imports
from __future__ import print_function

import argparse
import time
import os
import csv

import attr

import natnet
from scipy.spatial.transform import Rotation

file_extension = str(time.ctime().replace(' ', '-'))

# Other imports
import array as arr


@attr.s
class ClientApp(object):
    
    _client = attr.ib()
    _quiet = attr.ib()

    _last_printed = attr.ib(0)

    # optiData = attr.ib([])
    velocData = []
    target_rigid_body_id = 16
    _last_target_missing_printed = 0
    
    @classmethod
    def connect(cls, server_name, rate, quiet, target_rigid_body_id=None):
        # print('IN CONNECT')
        if server_name == 'fake':
            client = natnet.fakes.SingleFrameFakeClient.fake_connect(rate=rate)
        else:
            server_ip = server_name or os.getenv('OPTITRACK_SERVER_IP', '10.0.10.4')
            try:
                client = natnet.Client.connect(server_ip)
            except Exception as exc:
                print(f"OptiTrack connection failed for '{server_ip}': {exc}")
                client = None

        if client is None:
            return None
        
        app = cls(client, quiet)
        app.target_rigid_body_id = target_rigid_body_id
        return app

    def run(self):

        if self._quiet:
            self._client.set_callback(self.callback_quiet)
        else:
            self._client.set_callback(self.callback)
        
        
 
    def run_callback(self):
        self._client.run_once() # changed from .spin() to stop from constantly running

    def callback(self, rigid_bodies, markers, timing):
        """
        :type rigid_bodies: list[RigidBody]
        :type markers: list[LabelledMarker]
        :type timing: TimestampAndLatency
        """
        # print('IN CALLBACK')
        #print()
        #print('{:.1f}s: Received mocap frame'.format(timing.timestamp))
        self.optiData = [] # stores only the selected rigid body sample

        if rigid_bodies:
            selected_body = None

            for b in rigid_bodies:
                if b.id_ == self.target_rigid_body_id:
                    selected_body = b
                    break

            if selected_body is None:
                now = time.time()
                if now - self._last_target_missing_printed > 2:
                    print(f"Configured rigid body id {self.target_rigid_body_id} not present in this frame.")
                    self._last_target_missing_printed = now

            if selected_body is not None:
                self.optiData.append([time.time(), selected_body.id_, *(selected_body.position + selected_body.orientation)])

                       
        
        '''''
        if markers:
            print('Markers')
            for m in markers:
                print('\t Model {} marker {}: size {:.4f}mm, pos ({: 5.2f}, {: 5.2f}, {: 5.2f}), '.format(
                    m.model_id, m.marker_id, 1000*m.size, *m.position
                ))
        print('\t Latency: {:.1f}ms (system {:.1f}ms, transit {:.1f}ms, processing {:.2f}ms)'.format(
            1000*timing.latency, 1000*timing.system_latency, 1000*timing.transit_latency,
            1000*timing.processing_latency
        ))
        '''''

    def callback_quiet(self, *_):
        #print('IN CALLBACK QUIET')
        if time.time() - self._last_printed > 1:
            print('.')
            self._last_printed = time.time()
        return
    

class Optitrack:

    def __init__(self):
        self.optiCoordRange = 5
        self.optiCoordMin = 0
        self.optiAngRange = 360
        self.optiAngMin = -180 # converted to euler so -180 to 180
        self.last_valid_coord = [0.0, 0.0, 0.0]
        self.last_valid_angle = [0.0, 0.0, 0.0]

    
        parser = argparse.ArgumentParser()
        parser.add_argument('--server', help='Will autodiscover if not supplied')
        parser.add_argument('--fake', action='store_true',
                            help='Produce fake data at `rate` instead of connecting to actual server')
        parser.add_argument('--rate', type=float, default=10,
                            help='Rate at which to produce fake data (Hz)')
        parser.add_argument('--quiet', action='store_true')
        self.args, _unknown = parser.parse_known_args()

        self.target_rigid_body_id = 16

        folder = 'mocap-data'
        file_path = './' + folder + '/' + file_extension

        # Create experiment folder
        # if not os.path.exists(file_path):
        #    os.makedirs(file_path)

        
        # with open(os.path.join(file_path,
        #         'data.csv'), 'a') as fd:
        #     cwriter = csv.writer(fd)
        #     cwriter.writerow(['Timestamp', 'ID', 'Pos x', 'Pos y', 'Pos z', 'Quat w', 'Quat x', 'Quat y', 'Quat z']) # time.time() is time since 'epoch' - Jan 1 1970 00:00

        self.app = ClientApp.connect(
            'fake' if self.args.fake else self.args.server,
            self.args.rate,
            self.args.quiet,
            target_rigid_body_id=self.target_rigid_body_id,
        )

        if self.app is None:
            print(
                "Warning: Unable to connect to OptiTrack NatNet server. Set --server or OPTITRACK_SERVER_IP, "
                "or use --fake for offline testing. Position updates will use the last valid reading."
            )
            return


        self.app.run() # set up callback

    def optiTrackGetPos(self):
        try:
            if self.app is None:
                return self.last_valid_coord, self.last_valid_angle

            self.app.run_callback()
            frame_data = getattr(self.app, 'optiData', None)
            if not frame_data:
                return self.last_valid_coord, self.last_valid_angle

            sample = frame_data[0]
            if len(sample) < 9:
                return self.last_valid_coord, self.last_valid_angle

            coord = sample[2:5] # x y z
            normalizedOptiCoord = coord

            quaternion = sample[5:] # quat w, quat x, quat y, quat z
            quaternion = quaternion[1:] + quaternion[:1] # put into quat x, quat y, quat z, quat w form

            rotation = Rotation.from_quat(quaternion) # x y z w
            normalizedOptiAngle = rotation.as_euler('xzy', degrees=True)

            self.last_valid_coord = list(normalizedOptiCoord)
            self.last_valid_angle = list(normalizedOptiAngle)
            return normalizedOptiCoord, normalizedOptiAngle
        
        except (natnet.DiscoveryError, RuntimeError, IndexError, ValueError, AttributeError):
            return self.last_valid_coord, self.last_valid_angle



if __name__ == '__main__': # test
    opt = Optitrack()
    opt.optiTrackGetPos()
