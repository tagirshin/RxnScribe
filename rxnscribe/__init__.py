__version__ = "1.0.1"


def __getattr__(name):
    # Importing the ONNX runtime must not import the training stack.
    if name == "RxnScribe":
        from .interface import RxnScribe
        return RxnScribe
    if name == "RxnScribeONNX":
        from .onnx import RxnScribeONNX
        return RxnScribeONNX
    raise AttributeError(name)
