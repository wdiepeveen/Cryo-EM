import os
import sys
import inspect
import functools

class cached_property(object):
    def __init__(self, fun):
        self._fun = fun

    def __get__(self, obj, _):
        setattr(obj, self._fun.__name__, self._fun(obj))
        return getattr(obj, self._fun.__name__)

def suppress_stdout():
    sys.stdout = open(os.devnull, 'w')

def restore_stdout():
    sys.stdout = sys.__stdout__

def broadcast_io(indims, outdims):
    def decorator_func(func, indims=indims):
        params = [str(p) for p in inspect.signature(func).parameters.values()]
        ismethod = params[0] == "self"
        ninputs = len(params) - 1
        if ismethod:
            ninputs -= 1
        if type(indims) is not tuple:
            indims = (indims,)*ninputs
        @functools.wraps(func)
        def wrapped_func(*inputs, **kwargs):
            if ismethod:
                self = inputs[0]
                inputs = inputs[1:]
            assert len(inputs) == ninputs
            if len(kwargs.keys()) == 0:
                output = None
            else:
                assert len(kwargs.keys()) == 1
                output_name = list(kwargs.keys())[0]
                output = kwargs[output_name]
            multi = inputs[0].ndim - indims[0]
            inputs = list(inputs)
            if multi == 0:
                for i,el in enumerate(inputs):
                    assert el.ndim == indims[i]
                    inputs[i] = el[None,None]
                if output is not None:
                    assert output.ndim == outdims
                    output = output[(None,)*(ninputs + 1)]
            elif multi == 1:
                nbatch = inputs[0].shape[0]
                for i,el in enumerate(inputs):
                    assert el.ndim == indims[i] + 1
                    assert el.shape[0] == nbatch
                    inputs[i] = el[:,None]
                if output is not None:
                    assert output.ndim == outdims + 1
                    assert output.shape[0] == nbatch
                    output = output[(slice(None),) + (None,)*ninputs]
            else:
                nbatch = inputs[0].shape[0]
                for i,el in enumerate(inputs):
                    assert el.ndim == indims[i] + 2
                    assert el.shape[0] == nbatch
                if output is not None:
                    assert output.ndim == outdims + ninputs + 1
                    assert output.shape[0] == nbatch
            if output is not None:
                kwargs[output_name] = output
            if ismethod:
                inputs = (self,) + tuple(inputs)
            output = func(*inputs, **kwargs)
            if multi == 0:
                output = output[(0,)*(ninputs + 1)]
            elif multi == 1:
                output = output[(slice(None),) + (0,)*ninputs]
            return output
        return wrapped_func
    return decorator_func