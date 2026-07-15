"""Expected user-facing errors raised by the project CLI."""


class EBackboneV3Error(Exception):
    """Base class for actionable command/configuration errors."""


class ProbeError(EBackboneV3Error):
    """Raised when a real-data probe cannot be completed or verified."""


class SmokeError(EBackboneV3Error):
    """Raised when a synthetic execution smoke invariant fails."""


class SplitError(EBackboneV3Error):
    """Raised when supervised split manifests cannot be built or verified."""


class DatasetError(EBackboneV3Error):
    """Raised when a manifest-backed raw-event sample cannot be resolved safely."""
