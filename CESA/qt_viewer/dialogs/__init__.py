"""Qt-native dialogs for the CESA application."""

from .open_dataset_dialog import OpenDatasetDialog, OpenDatasetSelection
from .channel_mapping_dialog import ChannelMappingDialog, MappingResult
from .channel_selector_dialog import ChannelSelectorDialog
from .edf_import_wizard import EDFImportWizard

__all__ = [
    "OpenDatasetDialog",
    "OpenDatasetSelection",
    "ChannelMappingDialog",
    "MappingResult",
    "ChannelSelectorDialog",
    "EDFImportWizard",
]
