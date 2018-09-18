import os

class ModelLoader(object):
    @staticmethod
    def _import(attr, module_name):
        print("Loading {} from {} module...".format(attr, module_name))
        module = __import__(module_name, globals(), locals(), [], 1)
        return getattr(module, attr)

    @staticmethod
    def get_model(model_name):
        #'from lib.model import ***'
        return ModelLoader._import(model_name, 'model')