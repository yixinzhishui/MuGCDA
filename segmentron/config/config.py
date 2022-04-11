from __future__ import print_function
from __future__ import absolute_import     #https://blog.csdn.net/zzc15806/article/details/81133045
from __future__ import division
from __future__ import unicode_literals

import codecs
import yaml
import six
import time

from ast import literal_eval

class SegmentronConfig(dict):
    def __init__(self, *args, **kwargs):    #*args, **kwargs：https://www.cnblogs.com/zhangzhuozheng/p/8053045.html
        super(SegmentronConfig, self).__init__(*args, **kwargs)   #SegmentronConfig继承dict类，，将普通对象作为字典类(dict)使用,,字典参数传进来后，初始化dict类，，https://blog.csdn.net/jiangxiaoma111/article/details/39212835?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.control
        self.immutable = False

    def __setattr__(self, key, value, create_if_not_exist=True):   #def __setattr__：凡是成员变量赋值都会进入这个方法：https://blog.csdn.net/yusuiyu/article/details/87945149?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.control
        if key in ["immutable"]:
            self.__dict__[key] = value   #使用self.__dict__可避开死循环，避免循环调用_对象的__dict__中存储了一些self.xxx的一些东西_setattr__方法   #https://www.cnblogs.com/alvin2010/p/9102344.html
            return     #在创建类实例时会进入此if语句

        t = self         #t:类的所有成员变量（字典形式），，SegmentronConfig当做字典类（dict）使用，
        keylist = key.split(".")
        for k in keylist[:-1]:
            t = t.__getattr__(k, create_if_not_exist)

        t.__getattr__(keylist[-1], create_if_not_exist)
        t[keylist[-1]] = value   #更新现有字典对应属性的值

    def __getattr__(self, key, create_if_not_exist=True):
        if key in ["immutable"]:
            if key not in self.__dict__:
                self.__dict__[key] = False
            return self.__dict__[key]

        if not key in self:       #当键值在self的成员变量(现有字典（cfg实例）)中找不到，，
            if not create_if_not_exist:
                raise KeyError
            self[key] = SegmentronConfig()   #SegmentronConfig()继承dict()类   语义分割配置树中每一个节点都是一个SegmentronConfig()对象
        return self[key]     # 现有字典存在此键值，，返回对应的值（可能还是字典）

    def __setitem__(self, key, value):       #__setitem__：https://www.kancloud.cn/tksj/learn_python/491308  #https://www.nhooo.com/note/qa5cy6.html
        #
        if self.immutable:
            raise AttributeError(
                'Attempted to set "{}" to "{}", but SegConfig is immutable'.
                format(key, value))
        #
        if isinstance(value, six.string_types):
            try:
                value = literal_eval(value)
            except ValueError:
                pass
            except SyntaxError:
                pass
        super(SegmentronConfig, self).__setitem__(key, value)

    def update_from_other_cfg(self, other):
        if isinstance(other, dict):
            other = SegmentronConfig(other)  #SegmentronConfig继承dict类，将普通对象当做字典类(dict)使用
        assert isinstance(other, SegmentronConfig)
        cfg_list = [("", other)]
        while len(cfg_list):      #一层一层遍历yaml表示的字典                        #遍历文件yaml（以字典形式存在）中的元素，，并赋值给SegmentronConfig对象中的元素
            prefix, tdic = cfg_list[0]
            cfg_list = cfg_list[1:]    #cfg_list:[]
            for key, value in tdic.items():
                key = "{}.{}".format(prefix, key) if prefix else key
                if isinstance(value, dict):       #value为字典（dict）类型，，加入cfg_list，继续遍历
                    cfg_list.append((key, value))
                    continue
                try:
                    self.__setattr__(key, value, create_if_not_exist=False)   #将yaml文件中的元素值赋值给SegmentronConfig类中的对应成员变量属性
                except KeyError:
                    raise KeyError('Non-existent config key: {}'.format(key))

    def remove_irrelevant_cfg(self):         #删除配置文件中除当前MODEL_NAME相关之外的其他MODEl元素
        model_name = self.MODEL.MODEL_NAME

        from ..models.model_zoo import MODEL_REGISTRY
        model_list = MODEL_REGISTRY.get_list()
        model_list_lower = [x.lower() for x in model_list]    #lower()：转换字符串中所有大写字母为小写字母：https://www.runoob.com/python/att-string-lower.html

        assert model_name.lower() in model_list_lower, "Expected model name in {}, but received {}"\
            .format(model_list, model_name)
        pop_keys = []
        for key in self.MODEL.keys():
            if key.lower() in model_list_lower:
                if model_name.lower() == 'pointrend' and \
                    key.lower() == self.MODEL.POINTREND.BASEMODEL.lower():
                    continue
            if key.lower() in model_list_lower and key.lower() != model_name.lower():
                pop_keys.append(key)
        for key in pop_keys:
            self.MODEL.pop(key)      #pop():删除列表中指定元素：https://www.w3school.com.cn/python/ref_list_pop.asp



    def check_and_freeze(self):
        self.TIME_STAMP = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
        # TODO: remove irrelevant config and then freeze
        self.remove_irrelevant_cfg()
        self.immutable = True

    def update_from_list(self, config_list):
        if len(config_list) % 2 != 0:
            raise ValueError(
                "Command line options config format error! Please check it: {}".
                format(config_list))
        for key, value in zip(config_list[0::2], config_list[1::2]):
            try:
                self.__setattr__(key, value, create_if_not_exist=False)
            except KeyError:
                raise KeyError('Non-existent config key: {}'.format(key))

    def update_from_file(self, config_file):
        with codecs.open(config_file, 'r', 'utf-8') as file:
            loaded_cfg = yaml.load(file, Loader=yaml.FullLoader)
        self.update_from_other_cfg(loaded_cfg)

    def set_immutable(self, immutable):
        self.immutable = immutable
        for value in self.values():
            if isinstance(value, SegmentronConfig):
                value.set_immutable(immutable)

    def is_immutable(self):
        return self.immutable