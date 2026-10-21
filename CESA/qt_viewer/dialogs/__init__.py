"""Qt-native dialogs for the CESA application."""

from .open_dataset_dialog import OpenDatasetDialog, OpenDatasetSelection
from .channel_mapping_dialog import ChannelMappingDialog, MappingResult
from .channel_selector_dialog import ChannelSelectorDialog
from .edf_import_wizard import EDFImportWizard
from .spectral_dialog import SpectralAnalysisDialog
from .stage_psd_dialog import StagePSDDialog
from .wavelet_spectrogram_dialog import WaveletSpectrogramDialog
from .batch_analysis_dialog import BatchAnalysisDialog
from .group_analysis_dialog import GroupAnalysisDialog
from .psg_report_dialog import PsgReportDialog
from .scoring_editor_dialog import ScoringEditorDialog
from .scoring_comparison_dialog import ScoringComparisonDialog
from .auto_scoring_config_dialog import AutoScoringConfigDialog
from .edf_segment_export_dialog import EdfSegmentExportDialog
from .markers_dialog import MarkersDialog
from .entropy_dialog import EntropyAnalysisDialog
from .temporal_analysis_dialog import TemporalAnalysisDialog
from .sleep_periods_dialog import SleepPeriodsDialog
from .artifact_detection_dialog import ArtifactDetectionDialog
from .ocular_cleaning_dialog import OcularCleaningDialog
from .connectivity_dialog import ConnectivityDialog
from .microstates_dialog import MicrostatesDialog
from .micro_events_dialog import MicroEventsDialog
from .coupling_plot_dialog import CouplingPlotDialog
from .specparam_dialog import SpecParamDialog
from .night_qc_dialog import NightQcDialog
from .ica_cleaning_dialog import IcaCleaningDialog
from .feature_search_dialog import FeatureSearchDialog

__all__ = [
    "OpenDatasetDialog",
    "OpenDatasetSelection",
    "ChannelMappingDialog",
    "MappingResult",
    "ChannelSelectorDialog",
    "EDFImportWizard",
    "SpectralAnalysisDialog",
    "StagePSDDialog",
    "WaveletSpectrogramDialog",
    "BatchAnalysisDialog",
    "GroupAnalysisDialog",
    "PsgReportDialog",
    "ScoringEditorDialog",
    "ScoringComparisonDialog",
    "AutoScoringConfigDialog",
    "EdfSegmentExportDialog",
    "MarkersDialog",
    "EntropyAnalysisDialog",
    "TemporalAnalysisDialog",
    "SleepPeriodsDialog",
    "ArtifactDetectionDialog",
    "OcularCleaningDialog",
    "ConnectivityDialog",
    "MicrostatesDialog",
    "MicroEventsDialog",
    "CouplingPlotDialog",
    "SpecParamDialog",
    "NightQcDialog",
    "IcaCleaningDialog",
    "FeatureSearchDialog",
]
