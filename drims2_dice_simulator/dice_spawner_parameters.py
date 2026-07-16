# flake8: noqa

# auto-generated DO NOT EDIT

from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import SetParametersResult
from rcl_interfaces.msg import FloatingPointRange, IntegerRange
from rclpy.clock import Clock
from rclpy.exceptions import InvalidParameterValueException
from rclpy.time import Time
import copy
import sys
import rclpy
import rclpy.parameter
from generate_parameter_library_py.python_validators import ParameterValidators



class dice_spawner_node:

    class Params:
        # for detecting if the parameter struct has been updated
        stamp_ = Time()

        face_up = 0
        dice_size = 0.027
        random_position = False
        x_min = 0.4
        x_max = 0.8
        y_min = -0.2
        y_max = 0.4
        surface_height = 0.0
        position = [0.6, 0.2, 0.0]



    class ParamListener:
        def __init__(self, node, prefix=""):
            self.prefix_ = prefix
            self.params_ = dice_spawner_node.Params()
            self.node_ = node
            self.logger_ = rclpy.logging.get_logger("dice_spawner_node." + prefix)

            self.declare_params()

            self.node_.add_on_set_parameters_callback(self.update)
            self.user_callback = None
            self.clock_ = Clock()

        def get_params(self):
            tmp = self.params_.stamp_
            self.params_.stamp_ = None
            paramCopy = copy.deepcopy(self.params_)
            paramCopy.stamp_ = tmp
            self.params_.stamp_ = tmp
            return paramCopy

        def is_old(self, other_param):
            return self.params_.stamp_ != other_param.stamp_

        def unpack_parameter_dict(self, namespace: str, parameter_dict: dict):
            """
            Flatten a parameter dictionary recursively.

            :param namespace: The namespace to prepend to the parameter names.
            :param parameter_dict: A dictionary of parameters keyed by the parameter names
            :return: A list of rclpy Parameter objects
            """
            parameters = []
            for param_name, param_value in parameter_dict.items():
                full_param_name = namespace + param_name
                # Unroll nested parameters
                if isinstance(param_value, dict):
                    nested_params = self.unpack_parameter_dict(
                            namespace=full_param_name + rclpy.parameter.PARAMETER_SEPARATOR_STRING,
                            parameter_dict=param_value)
                    parameters.extend(nested_params)
                else:
                    parameters.append(rclpy.parameter.Parameter(full_param_name, value=param_value))
            return parameters

        def set_params_from_dict(self, param_dict):
            params_to_set = self.unpack_parameter_dict('', param_dict)
            self.update(params_to_set)

        def set_user_callback(self, callback):
            self.user_callback = callback

        def clear_user_callback(self):
            self.user_callback = None

        def refresh_dynamic_parameters(self):
            updated_params = self.get_params()
            # TODO remove any destroyed dynamic parameters

            # declare any new dynamic parameters


        def update(self, parameters):
            updated_params = self.get_params()

            for param in parameters:
                if param.name == self.prefix_ + "face_up":
                    validation_result = ParameterValidators.bounds(param, 0, 6)
                    if validation_result:
                        return SetParametersResult(successful=False, reason=validation_result)
                    updated_params.face_up = param.value
                    self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))

                if param.name == self.prefix_ + "dice_size":
                    validation_result = ParameterValidators.bounds(param, 0.001, 1.0)
                    if validation_result:
                        return SetParametersResult(successful=False, reason=validation_result)
                    updated_params.dice_size = param.value
                    self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))

                if param.name == self.prefix_ + "random_position":
                    updated_params.random_position = param.value
                    self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))

                if param.name == self.prefix_ + "x_min":
                    updated_params.x_min = param.value
                    self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))

                if param.name == self.prefix_ + "x_max":
                    updated_params.x_max = param.value
                    self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))

                if param.name == self.prefix_ + "y_min":
                    updated_params.y_min = param.value
                    self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))

                if param.name == self.prefix_ + "y_max":
                    updated_params.y_max = param.value
                    self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))

                if param.name == self.prefix_ + "surface_height":
                    updated_params.surface_height = param.value
                    self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))

                if param.name == self.prefix_ + "position":
                    validation_result = ParameterValidators.fixed_size(param, 3)
                    if validation_result:
                        return SetParametersResult(successful=False, reason=validation_result)
                    updated_params.position = param.value
                    self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))



            updated_params.stamp_ = self.clock_.now()
            self.update_internal_params(updated_params)
            if self.user_callback:
                self.user_callback(self.get_params())
            return SetParametersResult(successful=True)

        def update_internal_params(self, updated_params):
            self.params_ = updated_params

        def declare_params(self):
            updated_params = self.get_params()
            # declare all parameters and give default values to non-required ones
            if not self.node_.has_parameter(self.prefix_ + "face_up"):
                descriptor = ParameterDescriptor(description=r"Face number facing upward (1–6, 0 = random)", read_only = False)
                descriptor.integer_range.append(IntegerRange())
                descriptor.integer_range[-1].from_value = 0
                descriptor.integer_range[-1].to_value = 6
                parameter = updated_params.face_up
                self.node_.declare_parameter(self.prefix_ + "face_up", parameter, descriptor)

            if not self.node_.has_parameter(self.prefix_ + "dice_size"):
                descriptor = ParameterDescriptor(description=r"Length of the dice edge (in meters)", read_only = False)
                descriptor.floating_point_range.append(FloatingPointRange())
                descriptor.floating_point_range[-1].from_value = 0.001
                descriptor.floating_point_range[-1].to_value = 1.0
                parameter = updated_params.dice_size
                self.node_.declare_parameter(self.prefix_ + "dice_size", parameter, descriptor)

            if not self.node_.has_parameter(self.prefix_ + "random_position"):
                descriptor = ParameterDescriptor(description=r"Whether to spawn at a random position within bounds", read_only = False)
                parameter = updated_params.random_position
                self.node_.declare_parameter(self.prefix_ + "random_position", parameter, descriptor)

            if not self.node_.has_parameter(self.prefix_ + "x_min"):
                descriptor = ParameterDescriptor(description=r"Minimum X coordinate bounds", read_only = False)
                parameter = updated_params.x_min
                self.node_.declare_parameter(self.prefix_ + "x_min", parameter, descriptor)

            if not self.node_.has_parameter(self.prefix_ + "x_max"):
                descriptor = ParameterDescriptor(description=r"Maximum X coordinate bounds", read_only = False)
                parameter = updated_params.x_max
                self.node_.declare_parameter(self.prefix_ + "x_max", parameter, descriptor)

            if not self.node_.has_parameter(self.prefix_ + "y_min"):
                descriptor = ParameterDescriptor(description=r"Minimum Y coordinate bounds", read_only = False)
                parameter = updated_params.y_min
                self.node_.declare_parameter(self.prefix_ + "y_min", parameter, descriptor)

            if not self.node_.has_parameter(self.prefix_ + "y_max"):
                descriptor = ParameterDescriptor(description=r"Maximum Y coordinate bounds", read_only = False)
                parameter = updated_params.y_max
                self.node_.declare_parameter(self.prefix_ + "y_max", parameter, descriptor)

            if not self.node_.has_parameter(self.prefix_ + "surface_height"):
                descriptor = ParameterDescriptor(description=r"Surface height (in meters)", read_only = False)
                parameter = updated_params.surface_height
                self.node_.declare_parameter(self.prefix_ + "surface_height", parameter, descriptor)

            if not self.node_.has_parameter(self.prefix_ + "position"):
                descriptor = ParameterDescriptor(description=r"Initial dice position [x, y, z]", read_only = False)
                parameter = updated_params.position
                self.node_.declare_parameter(self.prefix_ + "position", parameter, descriptor)

            # TODO: need validation
            # get parameters and fill struct fields
            param = self.node_.get_parameter(self.prefix_ + "face_up")
            self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))
            validation_result = ParameterValidators.bounds(param, 0, 6)
            if validation_result:
                raise InvalidParameterValueException('face_up',param.value, 'Invalid value set during initialization for parameter face_up: ' + validation_result)
            updated_params.face_up = param.value
            param = self.node_.get_parameter(self.prefix_ + "dice_size")
            self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))
            validation_result = ParameterValidators.bounds(param, 0.001, 1.0)
            if validation_result:
                raise InvalidParameterValueException('dice_size',param.value, 'Invalid value set during initialization for parameter dice_size: ' + validation_result)
            updated_params.dice_size = param.value
            param = self.node_.get_parameter(self.prefix_ + "random_position")
            self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))
            updated_params.random_position = param.value
            param = self.node_.get_parameter(self.prefix_ + "x_min")
            self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))
            updated_params.x_min = param.value
            param = self.node_.get_parameter(self.prefix_ + "x_max")
            self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))
            updated_params.x_max = param.value
            param = self.node_.get_parameter(self.prefix_ + "y_min")
            self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))
            updated_params.y_min = param.value
            param = self.node_.get_parameter(self.prefix_ + "y_max")
            self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))
            updated_params.y_max = param.value
            param = self.node_.get_parameter(self.prefix_ + "surface_height")
            self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))
            updated_params.surface_height = param.value
            param = self.node_.get_parameter(self.prefix_ + "position")
            self.logger_.debug(param.name + ": " + param.type_.name + " = " + str(param.value))
            validation_result = ParameterValidators.fixed_size(param, 3)
            if validation_result:
                raise InvalidParameterValueException('position',param.value, 'Invalid value set during initialization for parameter position: ' + validation_result)
            updated_params.position = param.value


            self.update_internal_params(updated_params)
