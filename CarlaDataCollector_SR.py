#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Sensor synchronization example for CARLA

The communication model for the syncronous mode in CARLA sends the snapshot
of the world and the sensors streams in parallel.
We provide this script as an example of how to syncrononize the sensor
data gathering in the client.
To to this, we create a queue that is being filled by every sensor when the
client receives its data and the main loop is blocked until all the sensors
have received its data.
This suppose that all the sensors gather information at every tick. It this is
not the case, the clients needs to take in account at each frame how many
sensors are going to tick at each frame.
"""

from email.contentmanager import raw_data_manager
import glob
import os
import sys
from queue import Queue
from queue import Empty
from numpy import random
import time
from PIL import Image
import numpy as np
import cv2

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import VehicleLightState as vls



start_value=9500
counter_cam=start_value
counter_seg=start_value
counter_opt=start_value
counter_depth=start_value

# Sensor callback.
# This is where you receive the sensor data and
# process it as you liked and the important part is that,
# at the end, it should include an element into the sensor queue.
def sensor_callback_opt(sensor_data, sensor_queue, sensor_name,image):
    global counter_opt
    image = sensor_data.get_color_coded_flow()
    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    img[:,:,3] = 255
    pil_image = Image.fromarray(img)
    pil_image.save(f'ImagesCarla/imgs_train/OpticalFlow/Img_OpticalFlow-{counter_opt:d}.png')
    # Then you just need to add to the queue
    sensor_queue.put((sensor_data.frame, sensor_name))
    counter_opt += 1

def sensor_callback_cam(sensor_queue,image):
    global counter_cam
    image.save_to_disk(f'ImagesCarla\imgs_train\High_resolution/Img_HR-{counter_cam:d}.png')
    sensor_queue.put((image.frame, "camera"))
    counter_cam += 1
    
def sensor_callback_seg(sensor_queue,image):
    global counter_seg
    image.convert(carla.ColorConverter.CityScapesPalette)
    img_bgr = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    img_bgr[:,:,3] = 0
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_image_seg = Image.fromarray(img_rgb)
    pil_image_seg.save(f'ImagesCarla/imgs_train/Segmentation/Img_Segmentation-{counter_seg:d}.png')
    sensor_queue.put((image.frame, "segmentation_sensor"))
    counter_seg += 1


def sensor_callback_depth(sensor_queue,image):
    global counter_depth
    image.convert(carla.ColorConverter.LogarithmicDepth)
    image.save_to_disk(f'ImagesCarla\imgs_train\Depth/Img_depth-{counter_depth:d}.png')
    sensor_queue.put((image.frame, "camera_depth"))
    counter_depth += 1

def main():
    # Connect to the client and get the world object
    client = carla.Client('localhost', 2000)
    Towns=['Town01','Town02','Town03','Town04','Town05']
    #town=Towns[random.randint(1, 4)]
    #print(town)
    town='Town04'
    world = client.load_world(town)

    try:
        # We need to save the settings to be able to recover them at the end
        # of the script to leave the server in the same state that we found it.
        original_settings = world.get_settings()
        settings = world.get_settings()

        # We set CARLA syncronous mode
        settings.fixed_delta_seconds = 0.2 #5 frames per second 
        settings.synchronous_mode = True
        world.apply_settings(settings)

        # We create the sensor queue in which we keep track of the information
        # already received. This structure is thread safe and can be
        # accessed by all the sensors callback concurrently without problem.
        sensor_queue = Queue()

        # Bluepints for the sensors
        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points() 

        # Get the blueprint for the vehicle you want
        vehicle_bp = blueprint_library.find('vehicle.lincoln.mkz_2020') 

        # Try spawning the vehicle at a randomly chosen spawn point
        vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

        vehicle.set_autopilot(True)
        # Move the spectator behind the vehicle 
        spectator = world.get_spectator() 
        transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)),vehicle.get_transform().rotation) 
        spectator.set_transform(transform) 

        # Add traffic to the simulation
        for i in range(20): 
            vehicle_bp = random.choice(blueprint_library.filter('vehicle')) 
            npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points)) 

        # Set all vehicles in motion using the Traffic Manager
        for v in world.get_actors().filter('*vehicle*'): 
            v.set_autopilot(True) 
        #set the weather
        presets=[
        carla.WeatherParameters.ClearNoon,
        carla.WeatherParameters.CloudyNoon,
        carla.WeatherParameters.WetNoon,
        carla.WeatherParameters.WetCloudyNoon,
        carla.WeatherParameters.MidRainyNoon,
        carla.WeatherParameters.HardRainNoon,
        carla.WeatherParameters.SoftRainNoon,
        carla.WeatherParameters.ClearSunset,
        carla.WeatherParameters.CloudySunset,
        carla.WeatherParameters.WetSunset,
        carla.WeatherParameters.WetCloudySunset,
        carla.WeatherParameters.MidRainSunset,
        carla.WeatherParameters.HardRainSunset,
        carla.WeatherParameters.SoftRainSunset]
        #weather = carla.WeatherParameters(
           # cloudyness=80.0,
          #  precipitation=30.0,
           # sun_altitude_angle=70.0)
        #world.set_weather(weather)
        #print(world.get_weather())
        world.set_weather(presets[random.randint(0, 13)])
        
        # We create all the sensors and keep them in a list for convenience.
        sensor_list = []
        
        Low_resolution = (320, 188)
        High_resolution = (1920, 1080)

        camera_hr_bp = blueprint_library.find('sensor.camera.rgb') 
        camera_hr_bp.set_attribute('image_size_x', str(High_resolution[0])) 
        camera_hr_bp.set_attribute('image_size_y', str(High_resolution[1]))
        camera_hr_init_trans = carla.Transform(carla.Location(x=1.5,z=2))
        camera_hr = world.spawn_actor(camera_hr_bp, camera_hr_init_trans, attach_to=vehicle)
        camera_hr.listen(lambda image: sensor_callback_cam(sensor_queue,image))
        sensor_list.append(camera_hr)
         

        segmentation_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        segmentation_bp.set_attribute('image_size_x', str(High_resolution[0])) 
        segmentation_bp.set_attribute('image_size_y', str(High_resolution[1]))
        segmentation_transform = carla.Transform(carla.Location(x=1.5, z=2))
        segmentation_sensor = world.spawn_actor(segmentation_bp, segmentation_transform, attach_to=vehicle)
        segmentation_sensor.listen(lambda image: sensor_callback_seg(sensor_queue, image))
        sensor_list.append(segmentation_sensor)

        # Création du capteur de flux optique
        optical_flow_bp = blueprint_library.find('sensor.camera.optical_flow')
        optical_flow_bp.set_attribute('image_size_x', str(High_resolution[0])) 
        optical_flow_bp.set_attribute('image_size_y', str(High_resolution[1]))
        optical_flow_transform = carla.Transform(carla.Location(x=1.5, z=2))
        optical_flow_sensor = world.spawn_actor(optical_flow_bp, optical_flow_transform, attach_to=vehicle)
        optical_flow_sensor.listen(lambda data, flow_image=optical_flow_sensor: sensor_callback_opt(data, sensor_queue, "optical_flow_sensor", flow_image))
        sensor_list.append(optical_flow_sensor)


        camera_depth_bp = blueprint_library.find('sensor.camera.depth') 
        camera_depth_bp.set_attribute('image_size_x', str(High_resolution[0])) 
        camera_depth_bp.set_attribute('image_size_y', str(High_resolution[1]))
        camera_depth_init_trans = carla.Transform(carla.Location(x=1.5,z=2))
        camera_depth = world.spawn_actor(camera_depth_bp, camera_depth_init_trans, attach_to=vehicle)
        camera_depth.listen(lambda image: sensor_callback_depth(sensor_queue,image))
        sensor_list.append(camera_depth)

        # Main loop
        while True:
            # Tick the server
            world.tick()
            w_frame = world.get_snapshot().frame
            print("\nWorld's frame: %d" % w_frame)

            # Now, we wait to the sensors data to be received.
            # As the queue is blocking, we will wait in the queue.get() methods
            # until all the information is processed and we continue with the next frame.
            # We include a timeout of 1.0 s (in the get method) and if some information is
            # not received in this time we continue.
            try:
                for _ in range(len(sensor_list)):
                    s_frame = sensor_queue.get(True, 1.0)
                    print("    Frame: %d   Sensor: %s" % (s_frame[0], s_frame[1]))

            except Empty:
                print("    Some of the sensor information is missed")

    finally:
        world.apply_settings(original_settings)
        for sensor in sensor_list:
            sensor.destroy()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')
