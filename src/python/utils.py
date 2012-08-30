from scipy import signal
import numpy as np
import re
import copy_reg
import types
import warnings
import functools
warnings.simplefilter('always')


def fft_acf(data):
    '''Return the autocorrelation of a 1D array using the fft
    Note: the result is normalized'''
    data = data - np.mean(data)
    result = signal.fftconvolve(data, data[::-1])
    result = result[result.size / 2:] 
    return result / result[0]

def uneven_zip(*args):
    '''Zip the arguments together like the builtin function, except that
    when one argument runs out (because its shorter), you keep filling it in
    with its last value
    
    i.e.
    
    uneven_zip([1,2,3], 'a', [10,11]) = [[1, 'a', 10], [2, 'a', 11], [3, 'a', 11]]
    '''
    num_args = len(args)
    args = list(args)
    for i in xrange(num_args):
        if not hasattr(args[i], '__len__'):
            args[i] = (args[i],)
    lengths = [len(arg) for arg in args]
    def get(i):
        result = [None] * num_args
        for j in range(num_args):
            try:
                result[j] = args[j][i]
            except:
                result[j] = args[j][lengths[j]-1]
        return result
    zipped = map(get, range(max(lengths)))
    return zipped



def format_block(block):
    '''Format the given block of text, trimming leading/trailing
    empty lines and any leading whitespace that is common to all lines.
    The purpose is to let us list a code block as a multiline,
    triple-quoted Python string, taking care of indentation concerns.'''
    # separate block into lines
    lines = str(block).split('\n')
    # remove leading/trailing empty lines
    while lines and not lines[0]:
        del lines[0]
    while lines and not lines[-1]:
        del lines[-1]
    # look at first line to see how much indentation to trim
    ws = re.match(r'\s*', lines[0]).group(0)
    if ws:
            lines = map( lambda x: x.replace(ws,'',1), lines )
    # remove leading/trailing blank lines (after leading ws removal)
    # we do this again in case there were pure-whitespace lines
    while lines and not lines[0]:
        del lines[0]
    while lines and not lines[-1]:
        del lines[-1]
    
    return '\n'.join(lines)+'\n'


def keynat(string):
    '''A natural sort helper function for sort() and sorted()
    without using regular expression.

    >>> items = ('Z', 'a', '10', '1', '9')
    >>> sorted(items)
    ['1', '10', '9', 'Z', 'a']
    >>> sorted(items, key=keynat)
    ['1', '9', '10', 'Z', 'a']
    '''
    r = []
    for c in string:
        try:
            c = int(c)
            try:
                r[-1] = r[-1] * 10 + c
            except:
                r.append(c)
        except:
            r.append(c)
    return r


def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

def make_methods_pickable():
    "Run this at the top of a script to register pickable methods"
    copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)


def deprecated(replacement=None, removal_version=None):
    """A decorator which can be used to mark functions as deprecated.
    replacement is a callable that will be called with the same args
    as the decorated function.
    
    Code adapted from http://code.activestate.com/recipes/577819-deprecated-decorator/,
    MIT license

    >>> @deprecated()
    ... def foo(x):
    ...     return x
    ...
    >>> ret = foo(1)
    DeprecationWarning: foo is deprecated
    >>> ret
    1
    >>>
    >>>
    >>> def newfun(x):
    ...     return 0
    ...
    >>> @deprecated(newfun)
    ... def foo(x):
    ...     return x
    ...
    >>> ret = foo(1)
    DeprecationWarning: foo is deprecated; use newfun instead
    >>> ret
    0
    >>>
    """
    def outer(oldfun):
        def inner(*args, **kwargs):
            msg = "%s is deprecated use %s instead" % (oldfun.__name__, replacement.__name__)

            if removal_version is not None:
                msg += '%s will be removed in version %s' % (oldfun.__name__, removal_version)
                
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            
            return replacement(*args, **kwargs)

        return inner
    return outer

def future_warning(func):
    '''This is a decorator which can be used to mark functions
    as to-be deprecated. It will result in a warning being emitted
    when the function is used.'''

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.warn_explicit(
            "Call to future function {}.".format(func.__name__),
            category=FutureWarning,
            filename=func.func_code.co_filename,
            lineno=func.func_code.co_firstlineno + 1
        )
        return func(*args, **kwargs)
    return new_func