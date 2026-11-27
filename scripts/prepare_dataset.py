"""Rebuild the bundled full dataset from a checksum-pinned public UCI mirror.

Run from the repository root: python scripts/prepare_dataset.py [local-source.csv]
No source rows, values, or categories are removed. Only the target header is renamed.
"""
from pathlib import Path
from urllib.request import urlopen
import gzip
import hashlib
import io
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = ('https://raw.githubusercontent.com/MatteoM95/'
    'Default-of-Credit-Card-Clients-Dataset-Analisys/'
    '585f77c33dff5a14dac8e5396bec24820d3db2f8/dataset/default_of_credit_card_clients.csv')
SOURCE_SHA256 = '4edce6da569b2c30788a85fd98a9bafb1a983e8ca1a7aac875dd7ff853cbd0bb'
WORKBOOK_SHA256 = 'c76095b2a65c6e1b0ef6bdfab785346065b02f3f50cab6751d5a46b4c01c9e44'


def prepare(source=None):
    content = Path(source).read_bytes() if source else urlopen(SOURCE_URL, timeout=60).read()
    if hashlib.sha256(content).hexdigest() != SOURCE_SHA256:
        raise ValueError('Source checksum mismatch; do not use an unverified replacement.')
    frame = pd.read_csv(io.BytesIO(content))
    if frame.shape != (30000, 25) or frame.iloc[:, -1].value_counts().to_dict() != {0: 23364, 1: 6636}:
        raise ValueError('Source is not the expected complete UCI dataset.')
    frame = frame.rename(columns={'default payment next month': 'default_payment_next_month'})
    bundled = gzip.compress(frame.to_csv(index=False, lineterminator='\n').encode(), mtime=0)
    destination = ROOT / 'data'
    destination.mkdir(exist_ok=True)
    (destination / 'credit_card_default.csv.gz').write_bytes(bundled)
    metadata = {'name': 'Default of Credit Card Clients', 'creator': 'I-Cheng Yeh', 'year': 2009,
        'uci_url': 'https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients',
        'doi': 'https://doi.org/10.24432/C55S3H', 'license': 'CC BY 4.0',
        'license_url': 'https://creativecommons.org/licenses/by/4.0/',
        'source_mirror_url': SOURCE_URL, 'source_csv_sha256': SOURCE_SHA256,
        'original_workbook_sha256': WORKBOOK_SHA256,
        'bundled_sha256': hashlib.sha256(bundled).hexdigest(),
        'rows': 30000, 'predictors': 23, 'identifier': 'ID', 'target': 'default_payment_next_month',
        'class_counts': {'0': 23364, '1': 6636}, 'compressed_bytes': len(bundled),
        'changes': ['Target header renamed to default_payment_next_month; all 30,000 records and original values retained.']}
    (destination / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Bundled all {len(frame):,} rows; compressed size {len(bundled):,} bytes.')


if __name__ == '__main__':
    prepare(sys.argv[1] if len(sys.argv) > 1 else None)
