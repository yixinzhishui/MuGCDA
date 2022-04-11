# this code heavily based on detectron2

import os
import logging
import torch

from ..config import cfg


class Registry(object):
    """
    The registry that provides name -> object mapping, to support third-party users' custom modules.

    To create a registry (inside segmentron):

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:

    .. code-block:: python

        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...

    Or:

    .. code-block:: python

        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name):
        """
        Args:
            name (str): the name of this registry
        """
        self._name = name

        self._obj_map = {}

    def _do_register(self, name, obj):
        assert (
            name not in self._obj_map
        ), "An object named '{}' was already registered in '{}' registry!".format(name, self._name)
        self._obj_map[name] = obj
                                                   #可在@修饰符前打断点调试，印证其原理
    def register(self, obj=None, name=None):       #@函数修饰符： https://www.cnblogs.com/gdjlc/p/11182441.html  常用类修饰符：https://www.cnblogs.com/shuzf/p/11649339.html
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class, name=name):
                if name is None:
                    name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return deco     #register为修饰符，返回一个新函数(此函数再将类不做改变的返回去，只增加了一个注册过程：self._do_register(name, func_or_class))
                            #修饰符register带参数obj和name， name作为deco的参数传入，也可以直接调用name
                            #当修饰符带参数(obj, name)时，被修饰函数或类(比如DeepLabV3Plus)，会作为参数传入修饰符函数(register)的子函数(deco)的形式参数(func_or_class)
                            #当修饰符不带参数时，被修饰符或类，作为修饰符的形式参数，，，https://www.cnblogs.com/gdjlc/p/11182441.html有例子
        # used as a function call
        if name is None:
            name = obj.__name__    #类名：https://www.runoob.com/python/python-object.html
        self._do_register(name, obj)



    # def get(self, name):
    #     def load_backbone_pretrained(model):
    #         if cfg.PHASE == 'train' and cfg.TRAIN.BACKBONE_PRETRAINED and (not cfg.TRAIN.PRETRAINED_MODEL_PATH):
    #             if os.path.isfile(cfg.TRAIN.BACKBONE_PRETRAINED_PATH):
    #                 logging.info('Load backbone pretrained model from {}'.format(
    #                     cfg.TRAIN.BACKBONE_PRETRAINED_PATH
    #                 ))
    #                 msg = model.encoder.load_state_dict(torch.load(cfg.TRAIN.BACKBONE_PRETRAINED_PATH), strict=False)
    #                 logging.info(msg)
    #
    #     if name == "efficient_unet":
    #         ret = smp.Unet(encoder_name=cfg.MODEL.BACKBONE, in_channels=3, classes=cfg.DATASET.NUM_CLASSES, encoder_weights=None)
    #         #load_backbone_pretrained(ret)
    #     else:
    #         ret = self._obj_map.get(name)
    #         if ret is None:
    #             raise KeyError("No object named '{}' found in '{}' registry!".format(name, self._name))
    #
    #     return ret

    def get(self, name):
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError("No object named '{}' found in '{}' registry!".format(name, self._name))

        return ret

    def get_list(self):
        return list(self._obj_map.keys())
