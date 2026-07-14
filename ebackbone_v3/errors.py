"""Expected user-facing errors raised by the project CLI."""


class EBackboneV3Error(Exception):
    """Base class for actionable command/configuration errors."""


class ProbeError(EBackboneV3Error):
    """Raised when a real-data probe cannot be completed or verified."""


class SmokeError(EBackboneV3Error):
    """Raised when a synthetic execution smoke invariant fails."""

