import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose


@given(
    min_gene=st.floats(allow_nan=False, min_value=-30.0, max_value=10.0),
    len_gene=st.floats(allow_nan=False, min_value=0.1, max_value=20),
    min_value=st.floats(allow_nan=False, min_value=-1e5, max_value=1e5),
    len_value=st.floats(allow_nan=False, min_value=1e-2, max_value=1e5),
)
def test_linear_transform(min_gene, len_gene, min_value, len_value):
    from pygad_utils import LinearTransform

    max_gene = min_gene + len_gene
    max_value = min_value + len_value

    transformer = LinearTransform((min_gene, max_gene), (min_value, max_value))

    assert_allclose(min_value, transformer.to_value(min_gene), atol=1e-5)
    assert_allclose(max_value, transformer.to_value(max_gene), atol=1e-5)

    assert_allclose(min_gene, transformer.to_gene(min_value), atol=1e-5)
    assert_allclose(max_gene, transformer.to_gene(max_value), atol=1e-5)

    mid_gene = 0.5 * (min_gene + max_gene)
    mid_value = 0.5 * (min_value + max_value)
    assert_allclose(mid_value, transformer.to_value(mid_gene), atol=1e-5)
    assert_allclose(mid_gene, transformer.to_gene(mid_value), atol=1e-5)


@given(
    min_gene=st.floats(allow_nan=False, min_value=-30.0, max_value=10.0),
    len_gene=st.floats(allow_nan=False, min_value=0.1, max_value=20),
    min_value=st.floats(allow_nan=False, min_value=-1e5, max_value=1e5),
    len_value=st.floats(allow_nan=False, min_value=1e-2, max_value=1e5),
)
def test_exponential_transform(min_gene, len_gene, min_value, len_value):
    from pygad_utils import ExponentialTransform

    max_gene = min_gene + len_gene
    max_value = min_value + len_value

    transformer = ExponentialTransform((min_gene, max_gene), (min_value, max_value))

    assert_allclose(min_value, transformer.to_value(min_gene), atol=1e-5)
    assert_allclose(max_value, transformer.to_value(max_gene), atol=1e-5)

    assert_allclose(min_gene, transformer.to_gene(min_value), atol=1e-5)
    assert_allclose(max_gene, transformer.to_gene(max_value), atol=1e-5)

    mid_gene = 0.5 * (min_gene + max_gene)
    mid_value = min_value * np.sqrt(max_value / min_value)
    assert_allclose(mid_value, transformer.to_value(mid_gene), atol=1e-5)
    assert_allclose(mid_gene, transformer.to_gene(mid_value), atol=1e-5)
