# This is title of the package

## Introduction
Write something

## Struture
Main package is `code_preprocessing` where storing all classes and functions. Other running files (i.e. test files) will be included in `tests` folder.

## Usage
Running the command to install the package.
```bash
pipenv install -e .
```

In test files, include this line to import the package.
```python
from code_preprocessing.module_name import class, function
```

## Testing
Run the command
```bash
pytest/python tests/{test_file}.py
```