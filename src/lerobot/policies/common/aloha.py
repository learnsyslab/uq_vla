import torch


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with smolvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by smolvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


def pi_aloha_decode_state(state):
    # Flip the joints.
    for motor_idx in [1, 2, 8, 9]:
        state[:, motor_idx] *= -1
    # Reverse the gripper transformation that is being applied by the Aloha runtime.
    for motor_idx in [6, 13]:
        state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
    return state

def pi_aloha_encode_actions(actions):
    # Flip the joints.
    for motor_idx in [1, 2, 8, 9]:
        actions[:, :, motor_idx] *= -1
    # Reverse the gripper transformation that is being applied by the Aloha runtime.
    for motor_idx in [6, 13]:
        actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
    return actions

def pi_aloha_encode_actions_inv(actions):
    # Flip the joints again.
    for motor_idx in [1, 2, 8, 9]:
        actions[:, :, motor_idx] *= -1
    # Reverse the gripper transformation that is being applied by the Aloha runtime.
    for motor_idx in [6, 13]:
        actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
    return actions
