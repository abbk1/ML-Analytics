"""Bounded dataset loading, auditing, and explicitly authorized cleaning."""
from io import BytesIO
import hashlib
import numpy as np
import pandas as pd

MAX_UPLOAD_MB = 25
MAX_DATAFRAME_MB = 150
MAX_COLUMNS = 500


def load_dataset(content: bytes, filename: str, max_upload_mb: int = MAX_UPLOAD_MB) -> pd.DataFrame:
    if not content or len(content) > max_upload_mb * 1024**2:
        raise ValueError(f"Upload a nonempty file no larger than {max_upload_mb} MB.")
    suffix = filename.rsplit('.', 1)[-1].lower()
    if suffix in ('csv', 'tsv'):
        frame = pd.read_csv(BytesIO(content), sep='\t' if suffix == 'tsv' else ',', nrows=500001)
    elif suffix == 'xlsx':
        # Inspect workbook metadata before materializing large, compressed sheets.
        import openpyxl
        from zipfile import ZipFile
        with ZipFile(BytesIO(content)) as archive:
            if sum(item.file_size for item in archive.infolist()) > 200 * 1024**2:
                raise ValueError('Expanded workbook exceeds the 200 MB safety limit.')
        book = openpyxl.load_workbook(BytesIO(content), read_only=True, data_only=True)
        try:
            sheet = book.worksheets[0]
            if (sheet.max_row or 0) > 500001 or (sheet.max_column or 0) > MAX_COLUMNS:
                raise ValueError('Workbook exceeds the row or column safety limit.')
            frame = pd.read_excel(BytesIO(content), engine='openpyxl', nrows=500001)
        finally:
            book.close()
    else:
        raise ValueError('Supported formats: CSV, TSV, and Excel .xlsx.')
    if frame.empty or len(frame) > 500000 or frame.shape[1] > MAX_COLUMNS:
        raise ValueError('Dataset must contain 1–500,000 rows and at most 500 columns.')
    frame.columns = frame.columns.map(str)
    if not frame.columns.is_unique:
        raise ValueError('Column names must be unique after conversion to strings.')
    if frame.memory_usage(deep=True).sum() > MAX_DATAFRAME_MB * 1024**2:
        raise ValueError(f'Dataset exceeds the {MAX_DATAFRAME_MB} MB in-memory safety limit.')
    return frame.reset_index(drop=True)


def dataset_fingerprint(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    digest.update(str(list(zip(frame.columns, map(str, frame.dtypes)))).encode())
    return digest.hexdigest()


def class_distribution(labels) -> pd.DataFrame:
    counts = pd.Series(labels).dropna().astype(str).value_counts().sort_index()
    return pd.DataFrame({'class': counts.index, 'count': counts.values,
                         'percentage': (counts / counts.sum() * 100).values})


def audit_dataset(frame: pd.DataFrame, target=None) -> dict:
    unique = frame.nunique(dropna=True)
    warnings = []
    identifiers = [c for c in frame if unique[c] == len(frame) and
                   ('id' in c.lower().split('_') or c.lower() in ('index', 'uuid'))]
    constants = unique[unique <= 1].index.tolist()
    if identifiers:
        warnings.append(f'Possible identifiers: {", ".join(identifiers)}. Consider excluding them.')
    if constants:
        warnings.append(f'Constant or empty columns: {", ".join(constants)}.')
    unsupported = [c for c in frame if pd.api.types.is_datetime64_any_dtype(frame[c])]
    if unsupported:
        warnings.append(f'Datetime features need explicit feature engineering before training: {unsupported}.')
    if len(frame) > 50000:
        warnings.append('Large dataset: training is capped at 50,000 rows; choose an explicit stratified sample.')
    if target and target in frame:
        if frame[target].isna().any():
            warnings.append('Missing target labels must be explicitly removed before training.')
        counts = frame[target].dropna().astype(str).value_counts()
        if not counts.empty and counts.min() < 6:
            warnings.append('Extremely rare classes may not support the requested split, CV, or SMOTE.')
        for col in frame.columns.drop(target):
            if frame[col].astype(str).equals(frame[target].astype(str)):
                warnings.append(f'Potential target leakage: {col} exactly matches the target.')
    summary = pd.DataFrame({'dtype': frame.dtypes.astype(str), 'missing': frame.isna().sum(),
                            'unique': unique, 'memory_bytes': frame.memory_usage(deep=True, index=False)})
    return {'rows': len(frame), 'features': len(frame.columns) - int(target in frame.columns if target else False),
            'missing': int(frame.isna().sum().sum()), 'duplicates': int(frame.duplicated().sum()),
            'memory_mb': float(frame.memory_usage(deep=True).sum() / 1024**2),
            'columns': summary, 'warnings': warnings}


def prepare_dataset(frame, config):
    if config.target not in frame or not config.features or config.target in config.features:
        raise ValueError('Choose a target and at least one distinct feature.')
    if len(set(config.features)) != len(config.features) or not set(config.features).issubset(frame.columns):
        raise ValueError('Selected features must be unique, existing columns.')
    chosen = frame.loc[:, config.features + [config.target]]
    messages = []
    if chosen[config.target].isna().any():
        if not config.preprocessing.drop_missing_target:
            raise ValueError('Target contains missing labels. Explicitly enable removal of missing targets.')
        before = len(chosen)
        chosen = chosen.dropna(subset=[config.target])
        messages.append(f'Removed {before - len(chosen)} rows with missing targets.')
    if config.preprocessing.drop_duplicates:
        before = len(chosen)
        chosen = chosen.drop_duplicates()
        messages.append(f'Removed {before - len(chosen)} duplicate selected records before splitting.')
    if config.preprocessing.drop_missing_features:
        before = len(chosen)
        chosen = chosen.dropna(subset=config.features)
        messages.append(f'Removed {before - len(chosen)} incomplete feature records before splitting.')
    if len(chosen) > config.max_rows:
        raise ValueError(f'Training limit is {config.max_rows:,} rows. Explicitly sample the dataset or use Colab.')
    X = chosen[config.features]
    y = chosen[config.target].astype(str)
    if y.nunique() < 2 or y.nunique() > 50:
        raise ValueError('Classification requires 2–50 target classes. Check that the target is categorical.')
    if y.value_counts().min() < 2:
        raise ValueError('Every class needs at least two observations for a stratified holdout split.')
    # Never resolve contradictory labels automatically.
    conflicting = chosen.groupby(config.features, dropna=False, observed=True)[config.target].nunique().gt(1).sum()
    if conflicting:
        messages.append(f'{conflicting} identical feature groups have conflicting labels; review data quality.')
    duplicate_features = int(X.duplicated().sum())
    if duplicate_features:
        messages.append(f'{duplicate_features} repeated feature records remain; holdout/CV may share duplicates. '
                        'Consider explicit deduplication or a group-aware validation design outside this app.')
    return X, y, messages


def stratified_sample(frame, target, rows, seed=42):
    from sklearn.model_selection import train_test_split
    if target not in frame or frame[target].isna().any():
        raise ValueError('Choose a target without missing labels before stratified sampling.')
    if not 2 * frame[target].nunique() <= rows < len(frame):
        raise ValueError('Sample size must be smaller than the dataset and at least twice the class count.')
    sample, _ = train_test_split(frame, train_size=int(rows), stratify=frame[target], random_state=seed)
    return sample.sort_index().reset_index(drop=True)
