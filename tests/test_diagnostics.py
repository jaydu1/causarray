import numpy as np
import pandas as pd
import pytest

from causarray import align_test_mask


def _reordered_results():
    return pd.DataFrame({
        'trt': ['pert_b', 'pert_a', 'pert_b', 'pert_a'],
        'gene_names': ['gene_2', 'gene_1', 'gene_1', 'gene_2'],
        'tau': [0.2, 0.1, -0.1, -0.2],
    })


def test_align_test_mask_uses_labels_not_result_row_order():
    results = _reordered_results()
    test_mask = np.array([[True, False], [False, True]])

    aligned = align_test_mask(
        results,
        test_mask,
        treatment_names=['pert_a', 'pert_b'],
        gene_names=['gene_1', 'gene_2'],
    )

    np.testing.assert_array_equal(aligned, [True, True, False, False])
    assert aligned.dtype == bool


def test_align_test_mask_accepts_named_dataframe_and_series():
    results = _reordered_results()
    frame = pd.DataFrame(
        [[True, False], [False, True]],
        index=['pert_a', 'pert_b'],
        columns=['gene_1', 'gene_2'],
    )
    series = frame.stack()

    from_frame = align_test_mask(results, frame)
    from_series = align_test_mask(results, series)

    np.testing.assert_array_equal(from_frame, [True, True, False, False])
    np.testing.assert_array_equal(from_series, from_frame)


def test_align_test_mask_supports_custom_result_columns():
    results = _reordered_results().rename(columns={
        'trt': 'perturbation', 'gene_names': 'gene',
    })
    frame = pd.DataFrame(
        [[True, False], [False, True]],
        index=['pert_a', 'pert_b'],
        columns=['gene_1', 'gene_2'],
    )

    aligned = align_test_mask(
        results, frame,
        treatment_col='perturbation', gene_col='gene',
    )

    np.testing.assert_array_equal(aligned, [True, True, False, False])


@pytest.mark.parametrize(
    'test_mask, kwargs, message',
    [
        (
            np.array([[1, 0], [0, 1]]),
            {'treatment_names': ['pert_a', 'pert_b'],
             'gene_names': ['gene_1', 'gene_2']},
            'Boolean',
        ),
        (
            np.array([True, False]),
            {'treatment_names': ['pert_a'], 'gene_names': ['gene_1', 'gene_2']},
            'two-dimensional',
        ),
        (
            np.ones((2, 1), dtype=bool),
            {'treatment_names': ['pert_a', 'pert_b'],
             'gene_names': ['gene_1', 'gene_2']},
            'shape',
        ),
    ],
)
def test_align_test_mask_rejects_malformed_arrays(test_mask, kwargs, message):
    with pytest.raises(ValueError, match=message):
        align_test_mask(_reordered_results(), test_mask, **kwargs)


def test_align_test_mask_rejects_missing_and_duplicate_tests():
    results = _reordered_results()
    incomplete = pd.Series(
        [True, False],
        index=pd.MultiIndex.from_tuples([
            ('pert_a', 'gene_1'), ('pert_a', 'gene_2'),
        ]),
    )
    duplicate_results = pd.concat([results, results.iloc[[0]]], ignore_index=True)
    duplicate_mask = pd.Series(
        [True, False],
        index=pd.MultiIndex.from_tuples([
            ('pert_a', 'gene_1'), ('pert_a', 'gene_1'),
        ]),
    )
    missing_label_mask = pd.DataFrame(
        [[True, False], [False, True]],
        index=['pert_a', None], columns=['gene_1', 'gene_2'],
    )

    with pytest.raises(ValueError, match='missing treatment-gene tests'):
        align_test_mask(results, incomplete)
    with pytest.raises(ValueError, match='duplicate treatment-gene tests'):
        align_test_mask(duplicate_results, incomplete)
    with pytest.raises(ValueError, match='duplicate treatment-gene tests'):
        align_test_mask(results.iloc[[1]], duplicate_mask)
    with pytest.raises(ValueError, match='must not be missing'):
        align_test_mask(results, missing_label_mask)


def test_align_test_mask_does_not_modify_results():
    results = _reordered_results()
    original = results.copy(deep=True)
    frame = pd.DataFrame(
        [[True, False], [False, True]],
        index=['pert_a', 'pert_b'],
        columns=['gene_1', 'gene_2'],
    )

    align_test_mask(results, frame)

    pd.testing.assert_frame_equal(results, original)
