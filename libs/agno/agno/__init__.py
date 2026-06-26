from importlib.metadata import PackageNotFoundError, version

from agno.exceptions import ApprovalRequired, CallDeferred, ToolApprovalRequired, ToolCallDeferred

try:
    __version__ = version("agno")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "ApprovalRequired",
    "CallDeferred",
    "ToolApprovalRequired",
    "ToolCallDeferred",
]
