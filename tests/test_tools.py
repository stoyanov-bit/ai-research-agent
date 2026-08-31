import pytest

from research_agent.tools import (
    calculator,
    analyze_csv,
    inspect_csv,
)


TEST_CSV = "data/experiment.csv"


def test_calculator_add():

    result = calculator(
        2,
        3,
        "add",
    )

    assert result == 5


def test_calculator_multiply():

    result = calculator(
        4,
        5,
        "multiply",
    )

    assert result == 20


def test_calculator_division():

    result = calculator(
        10,
        2,
        "divide",
    )

    assert result == 5


def test_calculator_division_by_zero():

    with pytest.raises(
        ValueError
    ):

        calculator(
            10,
            0,
            "divide",
        )


def test_calculator_unknown_operation():

    with pytest.raises(
        ValueError
    ):

        calculator(
            1,
            2,
            "power",
        )


def test_inspect_csv():

    result = inspect_csv(
        TEST_CSV
    )

    assert "snr" in result["columns"]

    assert "accuracy" in result["columns"]

    assert result["rows"] > 0


def test_analyze_csv_mean():

    result = analyze_csv(
        TEST_CSV,
        "mean",
        "snr",
    )

    assert isinstance(
        result,
        float,
    )


def test_analyze_csv_normalized_column():

    lower_case = analyze_csv(
        TEST_CSV,
        "mean",
        "snr",
    )

    upper_case = analyze_csv(
        TEST_CSV,
        "mean",
        "SNR",
    )

    assert lower_case == upper_case


def test_analyze_csv_missing_column():

    with pytest.raises(
        ValueError
    ):

        analyze_csv(
            TEST_CSV,
            "mean",
            "does_not_exist",
        )