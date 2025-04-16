from fastapi.testclient import TestClient

from libs.fastapi.settings import reset_settings_cache
from tests.utils.factory import get_create_app_func


def get_test_client(
        request,
        custom_config_path,
        monkeypatch,
        project_name: str,
        model_type: str
):
    use_gpu = request.param if hasattr(request, "param") else False

    # Override the environment variables so the app loads the custom config
    monkeypatch.setenv("USE_GPU", "true" if use_gpu else "false")
    monkeypatch.setenv("CONFIG_PATH", custom_config_path)
    monkeypatch.setenv("MAX_BATCH_SIZE", "8")

    # Reset the settings cache to ensure the new environment variable is picked up
    reset_settings_cache()

    app = get_create_app_func(project_name, model_type)()

    with TestClient(app) as c:
        # Create a variable to check which hardware is being used
        c.use_gpu = use_gpu

        yield c
