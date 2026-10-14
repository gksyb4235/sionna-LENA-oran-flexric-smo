"""Portable frozen AGORA runtime. No training workspace imports are required."""
from pathlib import Path
from .api import Runtime as LegacyRuntime


def Runtime(root=None, device='cpu'):
    selected=Path(root) if root else Path(__file__).parent/'assets_v16'
    if (selected/'manifest.json').exists() and (selected/'model.pt').exists():
        from .v16 import Runtime as App5Runtime
        return App5Runtime(selected,device)
    return LegacyRuntime(root,device)

__all__ = ["Runtime"]
