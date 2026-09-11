"""Safe provider failures that callers must not reinterpret as user intent."""


class ProviderUnavailableError(RuntimeError):
    """A transport/access failure, not malformed model output to retry."""

    def __init__(self, status_code=None):
        self.status_code = status_code
        if status_code in (401, 403):
            message = (
                "NVIDIA rejected access to the configured model (HTTP "
                f"{status_code}). Check the key and model access in your NVIDIA account. "
                "If needed, replace NVIDIA_API_KEY in your local .env file, then restart the app. "
                "Do not paste your key into chat. No tools were run."
            )
        elif status_code == 429:
            message = "NVIDIA's request limit was reached. Wait a moment before retrying. No tools were run."
        elif status_code == 404:
            message = "NVIDIA could not find the configured model. Check NVIDIA_MODEL in .env and restart the app. No tools were run."
        else:
            message = "Could not reach NVIDIA or receive a response. Check your connection and retry. No tools were run."
        super().__init__(message)
