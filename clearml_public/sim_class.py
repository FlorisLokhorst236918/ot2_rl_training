import pybullet as p
import time
import pybullet_data
import math
import logging
import os
import random

#logging.basicConfig(level=logging.INFO)

class Simulation:
    def __init__(self, num_agents, render=True, rgb_array=False):
        self.render = render
        self.rgb_array = rgb_array
        if render:
            mode = p.GUI # for graphical version
        else:
            mode = p.DIRECT # for non-graphical version
        # Set up the simulation
        self.physicsClient = p.connect(mode)
        # Hide the default GUI components
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.setGravity(0,0,-10)
        #p.setPhysicsEngineParameter(contactBreakingThreshold=0.000001)
        # load a texture
        texture_list = os.listdir("textures")
        random_texture = random.choice(texture_list[:-1])
        random_texture_index = texture_list.index(random_texture)
        self.plate_image_path = f'textures/_plates/{os.listdir("textures/_plates")[random_texture_index]}'
        self.textureId = p.loadTexture(f'textures/{random_texture}')
        #print(f'textureId: {self.textureId}')

        # Set the camera parameters
        cameraDistance = 1.1*(math.ceil((num_agents)**0.3)) # Distance from the target (zoom)
        cameraYaw = 90  # Rotation around the vertical axis in degrees
        cameraPitch = -35  # Rotation around the horizontal axis in degrees
        cameraTargetPosition = [-0.2, -(math.ceil(num_agents**0.5)/2)+0.5, 0.1]  # XYZ coordinates of the target position

        # Reset the camera with the specified parameters
        p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition)

        self.baseplaneId = p.loadURDF("plane.urdf")
        # add collision shape to the plane
        #p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[30, 305, 0.001])

        # define the pipette offset
        self.pipette_offset = [0.073, 0.0895, 0.0895]
        # dictionary to keep track of the current pipette position per robot
        self.pipette_positions = {}

        # Create the robots
        self.create_robots(num_agents)

        # list of sphere ids
        self.sphereIds = []

        # dictionary to keep track of the droplet positions on specimens key for specimenId, list of droplet positions
        self.droplet_positions = {}

    # method to create n robots in a grid pattern
    def create_robots(self, num_agents):
        spacing = 1  # Adjust the spacing as needed

        # Calculate the grid size to fit all agents
        grid_size = math.ceil(num_agents ** 0.5)

        self.robotIds = []
        self.specimenIds = []
        agent_count = 0  # Counter for the number of placed agents

        for i in range(grid_size):
            for j in range(grid_size):
                if agent_count < num_agents:  # Check if more agents need to be placed
                    # Calculate position for each robot
                    position = [-spacing * i, -spacing * j, 0.03]
                    robotId = p.loadURDF("ot_2_simulation_v6.urdf", position, [0,0,0,1],
                                        flags=p.URDF_USE_INERTIA_FROM_FILE)
                    start_position, start_orientation = p.getBasePositionAndOrientation(robotId)
                    p.createConstraint(parentBodyUniqueId=robotId,
                                    parentLinkIndex=-1,
                                    childBodyUniqueId=-1,
                                    childLinkIndex=-1,
                                    jointType=p.JOINT_FIXED,
                                    jointAxis=[0, 0, 0],
                                    parentFramePosition=[0, 0, 0],
                                    childFramePosition=start_position,
                                    childFrameOrientation=start_orientation)

                    # Load the specimen with an offset
                    offset = [0.18275-0.00005, 0.163-0.026, 0.057]
                    position_with_offset = [position[0] + offset[0], position[1] + offset[1], position[2] + offset[2]]
                    rotate_90 = p.getQuaternionFromEuler([0, 0, -math.pi/2])
                    planeId = p.loadURDF("custom.urdf", position_with_offset, rotate_90)#start_orientation)
                    # Disable collision between the robot and the specimen
                    p.setCollisionFilterPair(robotId, planeId, -1, -1, enableCollision=0)
                    spec_position, spec_orientation = p.getBasePositionAndOrientation(planeId)

                    p.createConstraint(parentBodyUniqueId=planeId,
                                    parentLinkIndex=-1,
                                    childBodyUniqueId=-1,
                                    childLinkIndex=-1,
                                    jointType=p.JOINT_FIXED,
                                    jointAxis=[0, 0, 0],
                                    parentFramePosition=[0, 0, 0],
                                    childFramePosition=spec_position,
                                    childFrameOrientation=spec_orientation)
                    # Load your texture and apply it to the plane
                    p.changeVisualShape(planeId, -1, textureUniqueId=self.textureId)

                    self.robotIds.append(robotId)
                    self.specimenIds.append(planeId)

                    agent_count += 1  # Increment the agent counter

                    # calculate the pipette position
                    pipette_position = self.get_pipette_position(robotId)
                    # save the pipette position
                    self.pipette_positions[f'robotId_{robotId}'] = pipette_position

    # method to get the current pipette position for a robot
    def get_pipette_position(self, robotId):
        #get the position of the robot
        robot_position = p.getBasePositionAndOrientation(robotId)[0]
        robot_position = list(robot_position)
        joint_states = p.getJointStates(robotId, [0, 1, 2])
        robot_position[0] -= joint_states[0][0]
        robot_position[1] -= joint_states[1][0]
        robot_position[2] += joint_states[2][0]
        # x,y offset
        x_offset = self.pipette_offset[0]
        y_offset = self.pipette_offset[1]
        z_offset = self.pipette_offset[2]
        # Calculate the position of the pipette at the tip of the pipette but at the same z coordinate as the specimen
        pipette_position = [robot_position[0]+x_offset, robot_position[1]+y_offset, robot_position[2]+z_offset]
        return pipette_position

    # method to reset the simulation
    def reset(self, num_agents=1):
        # Remove the textures from the specimens
        for specimenId in self.specimenIds:
            p.changeVisualShape(specimenId, -1, textureUniqueId=-1)

        # Remove the robots
        for robotId in self.robotIds:
            p.removeBody(robotId)
            # remove the robotId from the list of robotIds
            self.robotIds.remove(robotId)

        # Remove the specimens
        for specimenId in self.specimenIds:
            p.removeBody(specimenId)
            # remove the specimenId from the list of specimenIds
            self.specimenIds.remove(specimenId)

        # Remove the spheres
        for sphereId in self.sphereIds:
            p.removeBody(sphereId)
            # remove the sphereId from the list of sphereIds
            self.sphereIds.remove(sphereId)

        # dictionary to keep track of the current pipette position per robot
        self.pipette_positions = {}
        # list of sphere ids
        self.sphereIds = []
        # dictionary to keep track of the droplet positions on specimens key for specimenId, list of droplet positions
        self.droplet_positions = {}

        # Create the robots
        self.create_robots(num_agents)

        return self.get_states()

    # method to run the simulation for a specified number of steps
    def run(self, actions, num_steps=1):
        start = time.time()
        n = 100
        for i in range(num_steps):
            self.apply_actions(actions)
            p.stepSimulation()

            # check contact for each robot and specimen
            for specimenId, robotId in zip(self.specimenIds, self.robotIds):
                self.check_contact(robotId, specimenId)

            if self.rgb_array:
                # Camera parameters
                camera_pos = [1, 0, 1] # Example position
                camera_target = [-0.3, 0, 0] # Point where the camera is looking at
                up_vector = [0, 0, 1] # Usually the Z-axis is up
                fov = 50 # Field of view
                aspect = 320/240 # Aspect ratio (width/height)

                # Get camera image
                width, height, rgbImg, depthImg, segImg = p.getCameraImage(width=320, height=240, viewMatrix=p.computeViewMatrix(camera_pos, camera_target, up_vector), projectionMatrix=p.computeProjectionMatrixFOV(fov, aspect, 0.1, 100.0))

                self.current_frame = rgbImg  # RGB array

            if self.render:
                time.sleep(1./240.) # slow down the simulation

        return self.get_states()

    # method to apply actions to the robots using velocity control
    def apply_actions(self, actions): # actions [[x,y,z,drop], [x,y,z,drop], ...
        for i in range(len(self.robotIds)):
            p.setJointMotorControl2(self.robotIds[i], 0, p.VELOCITY_CONTROL, targetVelocity=-actions[i][0], force=500)
            p.setJointMotorControl2(self.robotIds[i], 1, p.VELOCITY_CONTROL, targetVelocity=-actions[i][1], force=500)
            p.setJointMotorControl2(self.robotIds[i], 2, p.VELOCITY_CONTROL, targetVelocity=actions[i][2], force=800)
            if actions[i][3] == 1:
                self.drop(robotId=self.robotIds[i])

    # method to drop a simulated droplet on the specimen from the pipette
    def drop(self, robotId):
        robot_position = p.getBasePositionAndOrientation(robotId)[0]
        robot_position = list(robot_position)
        joint_states = p.getJointStates(robotId, [0, 1, 2])
        robot_position[0] -= joint_states[0][0]
        robot_position[1] -= joint_states[1][0]
        robot_position[2] += joint_states[2][0]
        # x,y offset
        x_offset = self.pipette_offset[0]
        y_offset = self.pipette_offset[1]
        z_offset = self.pipette_offset[2]-0.0015
        # Get the position of the specimen
        specimen_position = p.getBasePositionAndOrientation(self.specimenIds[0])[0]
        # Create a sphere to represent the droplet
        sphereRadius = 0.003  # Adjust as needed
        sphereColor = [1, 0, 0, 0.5]  # RGBA (Red in this case)
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=sphereColor)
        #add collision to the sphere
        collision = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=sphereRadius)
        sphereBody = p.createMultiBody(baseMass=0.1, baseVisualShapeIndex=visualShapeId, baseCollisionShapeIndex=collision)
        # Calculate the position of the droplet at the tip of the pipette but at the same z coordinate as the specimen
        droplet_position = [robot_position[0]+x_offset, robot_position[1]+y_offset, robot_position[2]+z_offset]
        p.resetBasePositionAndOrientation(sphereBody, droplet_position, [0, 0, 0, 1])
        # track the sphere id
        self.sphereIds.append(sphereBody)
        self.dropped = True
        return droplet_position

    # method to get the states of the robots
    def get_states(self):
        states = {}
        for robotId in self.robotIds:
            raw_joint_states = p.getJointStates(robotId, [0, 1, 2])

            # Convert joint states into a dictionary
            joint_states = {}
            for i, joint_state in enumerate(raw_joint_states):
                joint_states[f'joint_{i}'] = {
                    'position': joint_state[0],
                    'velocity': joint_state[1],
                    'reaction_forces': joint_state[2],
                    'motor_torque': joint_state[3]
                }

            # Robot position
            robot_position = p.getBasePositionAndOrientation(robotId)[0]
            robot_position = list(robot_position)

            # Adjust robot position based on joint states
            robot_position[0] -= raw_joint_states[0][0]
            robot_position[1] -= raw_joint_states[1][0]
            robot_position[2] += raw_joint_states[2][0]

            # Pipette position
            pipette_position = [robot_position[0] + self.pipette_offset[0],
                                robot_position[1] + self.pipette_offset[1],
                                robot_position[2] + self.pipette_offset[2]]
            # Round pipette position to 4 decimal places
            pipette_position = [round(num, 4) for num in pipette_position]

            # Store information in the dictionary
            states[f'robotId_{robotId}'] = {
                "joint_states": joint_states,
                "robot_position": robot_position,
                "pipette_position": pipette_position
            }

        return states

    # method to check contact with the spheres and the specimen and robot
    def check_contact(self, robotId, specimenId):
        for sphereId in self.sphereIds:
            # Check contact with the specimen
            contact_points_specimen = p.getContactPoints(sphereId, specimenId)
            # Check contact with the robot
            contact_points_robot = p.getContactPoints(sphereId, robotId)

            # If contact with the specimen is detected
            if contact_points_specimen:
                # Disable collision between the sphere and the specimen
                p.setCollisionFilterPair(sphereId, specimenId, -1, -1, enableCollision=0)
                # Get current position and orientation of the sphere
                sphere_position, sphere_orientation = p.getBasePositionAndOrientation(sphereId)
                # Fix the sphere in place relative to the world
                p.createConstraint(parentBodyUniqueId=sphereId,
                                    parentLinkIndex=-1,
                                    childBodyUniqueId=-1,
                                    childLinkIndex=-1,
                                    jointType=p.JOINT_FIXED,
                                    jointAxis=[0, 0, 0],
                                    parentFramePosition=[0, 0, 0],
                                    childFramePosition=sphere_position,
                                    childFrameOrientation=sphere_orientation)
                # track the final position of the sphere on the specimen
                if f'specimenId_{specimenId}' in self.droplet_positions:
                    self.droplet_positions[f'specimenId_{specimenId}'].append(sphere_position)
                else:
                    self.droplet_positions[f'specimenId_{specimenId}'] = [sphere_position]

            # If contact with the robot is detected
            if contact_points_robot:
                # Destroy the sphere
                p.removeBody(sphereId)
                # Remove the sphereId from the list of sphereIds
                self.sphereIds.remove(sphereId)

    def set_start_position(self, x, y, z):
        # Iterate through each robot and set its pipette to the start position
        for robotId in self.robotIds:
            # Adjust the x, y, z values based on the robot's current position and pipette offset
            robot_position = p.getBasePositionAndOrientation(robotId)[0]
            adjusted_x = x - robot_position[0] - self.pipette_offset[0]
            adjusted_y = y - robot_position[1] - self.pipette_offset[1]
            adjusted_z = z - robot_position[2] - self.pipette_offset[2]

            # Reset the joint positions/start position
            p.resetJointState(robotId, 0, targetValue=adjusted_x)
            p.resetJointState(robotId, 1, targetValue=adjusted_y)
            p.resetJointState(robotId, 2, targetValue=adjusted_z)

    # function to return the path of the current plate image
    def get_plate_image(self):
        return self.plate_image_path

    # close the simulation
    def close(self):
        p.disconnect()
