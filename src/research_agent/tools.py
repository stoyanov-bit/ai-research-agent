import pandas as pd

def calculator(a: float, b: float, operation: str) -> float:
    """Perform a basic arithmetic operation."""

    if operation == "add":
        return a + b

    if operation == "subtract":
        return a - b

    if operation == "multiply":
        return a * b

    if operation == "divide":
        if b == 0:
            raise ValueError("Division by zero is not allowed.")
        return a / b

    raise ValueError(
        f"Unknown operation: {operation}"
    )

def normalize_name(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def analyze_csv(
    file_path: str,
    operation: str,
    column: str | None = None,
):
    df = pd.read_csv(file_path)

    if operation == "summary":
        return df.describe(include="all").to_string()

    if column is None:
        raise ValueError(
            f"A column must be provided for {operation}."
        )

    normalized_columns = {
        normalize_name(col): col
        for col in df.columns
    }

    normalized_column = normalize_name(column)

    if normalized_column not in normalized_columns:
        raise ValueError(
            f"Column '{column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    actual_column = normalized_columns[normalized_column]

    if operation == "mean":
        return float(df[actual_column].mean())

    if operation == "median":
        return float(df[actual_column].median())

    if operation == "min":
        return float(df[actual_column].min())

    if operation == "max":
        return float(df[actual_column].max())

    raise ValueError(
        f"Unknown operation: {operation}"
    )

def inspect_csv(file_path: str) -> dict:
    """Return basic information about a CSV file."""

    df = pd.read_csv(file_path)

    return {
        "columns": list(df.columns),
        "rows": len(df),
        "dtypes": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },
    }