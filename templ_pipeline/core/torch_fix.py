# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Torch compatibility fix for Streamlit file watcher issue.
"""


def fix_torch_streamlit_compatibility():
    """Fix torch.classes path issue with Streamlit file watcher"""
    try:
        import torch

        torch.classes.__path__ = []
        return True
    except Exception:
        return False


# Apply fix immediately on import
fix_torch_streamlit_compatibility()
