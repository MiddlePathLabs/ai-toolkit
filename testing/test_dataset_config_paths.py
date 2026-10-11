"""DatasetConfig folder_path/dataset_path key contract tests.

The UI edits only folder_path; the data loader prefers dataset_path with
folder_path as fallback. Both keys set to different values means UI edits are
silently ignored, so that combination must fail fast at config parse time.
"""
from __future__ import annotations

import pytest

from toolkit.config_modules import DatasetConfig


def test_folder_path_only():
    cfg = DatasetConfig(folder_path='E:/data/subject')
    assert cfg.folder_path == 'E:/data/subject'
    assert cfg.dataset_path is None


def test_dataset_path_only():
    cfg = DatasetConfig(dataset_path='E:/data/subject')
    assert cfg.dataset_path == 'E:/data/subject'
    assert cfg.folder_path is None


def test_both_keys_same_path_mixed_separators_ok():
    # GUI-saved configs commonly mix \\ and / between the two keys
    cfg = DatasetConfig(
        folder_path='E:\\ai-toolkit\\AI-Toolkit\\datasets/subject',
        dataset_path='E:/ai-toolkit/AI-Toolkit/datasets/subject',
    )
    assert cfg.dataset_path == 'E:/ai-toolkit/AI-Toolkit/datasets/subject'


def test_both_keys_different_paths_raise():
    with pytest.raises(ValueError, match='folder_path'):
        DatasetConfig(
            folder_path='E:/ai-toolkit/AI-Toolkit/datasets/subject_v2',
            dataset_path='E:/ai-toolkit/AI-Toolkit/datasets/subject',
        )


def test_both_keys_case_difference_ok_on_windows_paths():
    # same folder typed with different case must not be treated as a conflict
    cfg = DatasetConfig(folder_path='E:/Data/Subject', dataset_path='e:/data/subject')
    assert cfg.folder_path == 'E:/Data/Subject'
