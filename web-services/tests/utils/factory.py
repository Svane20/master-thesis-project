def import_create_app_resnet_onnx():
    from resnet.onnx.main import create_app
    return create_app


def import_create_app_resnet_torchscript():
    from resnet.torchscript.main import create_app
    return create_app


def import_create_app_resnet_pytorch():
    from resnet.pytorch.main import create_app
    return create_app


def import_create_app_swin_onnx():
    from swin.onnx.main import create_app
    return create_app


def import_create_app_swin_torchscript():
    from swin.torchscript.main import create_app
    return create_app


def import_create_app_swin_pytorch():
    from swin.pytorch.main import create_app
    return create_app


def import_create_app_dpt_onnx():
    from dpt.onnx.main import create_app
    return create_app


def import_create_app_dpt_torchscript():
    from dpt.torchscript.main import create_app
    return create_app


def import_create_app_dpt_pytorch():
    from dpt.pytorch.main import create_app
    return create_app


APP_FACTORIES = {
    ("resnet", "onnx"): import_create_app_resnet_onnx,
    ("resnet", "torchscript"): import_create_app_resnet_torchscript,
    ("resnet", "pytorch"): import_create_app_resnet_pytorch,
    ("swin", "onnx"): import_create_app_swin_onnx,
    ("swin", "torchscript"): import_create_app_swin_torchscript,
    ("swin", "pytorch"): import_create_app_swin_pytorch,
    ("dpt", "onnx"): import_create_app_dpt_onnx,
    ("dpt", "torchscript"): import_create_app_dpt_torchscript,
    ("dpt", "pytorch"): import_create_app_dpt_pytorch,
}


def get_create_app_func(project_name: str, model_type: str):
    """
    Retrieves the create_app function for the given project_name and model_type.

    Raises ValueError if the combination is unrecognized.
    """
    key = (project_name, model_type)
    if key not in APP_FACTORIES:
        raise ValueError(f"No create_app function defined for {project_name} - {model_type}")
    return APP_FACTORIES[key]()
