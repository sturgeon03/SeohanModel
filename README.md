# SeohanModel
Seohan E-Corner vehicle dynamics and control simulation model

## Modeling Quick Start
- Install: `pip install -r requirements.txt`
- Run (built-in scenario): `python run_modeling_quickstart.py`
- List CBNU files: `python run_modeling_quickstart.py --list-cbnu-files`
- Run (CBNU replay): `python run_modeling_quickstart.py --input-source cbnu --cbnu-file 8`
- Outputs include PNG summaries and `trajectory.gif`
- Guide: `MODELING_QUICKSTART.md`

## EX Folder
- Scenario validation example: `python ex/run_scenario_validation.py --mode quick --duration 5`
- Guide: `ex/README.md`
