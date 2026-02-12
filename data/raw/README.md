# Raw Data

This directory contains original, immutable raw data files from medical databases and sources.

## Files

- **medical_records.csv** - Raw patient medical records (10,000+ records)
- **symptoms.csv** - Comprehensive symptom database
- **medicines.csv** - Medicine information database

## Data Source

Raw data sourced from:
- Medical institutions and hospitals
- Patient health records
- Clinical databases
- Public medical datasets

## Important Notes

1. **DO NOT MODIFY** - These files are immutable reference files
2. **Raw state** - Data includes missing values, duplicates, and inconsistencies
3. **Large files** - Files may exceed 50MB
4. **Use processed/ folder** - For cleaned and processed data

## Data Size

- medical_records.csv: ~30 MB
- symptoms.csv: ~15 MB
- medicines.csv: ~5 MB

Total: ~50 MB

## Processing Pipeline

Raw data → Cleaning → Feature Engineering → Processed data

See ../processed/ for output files.
