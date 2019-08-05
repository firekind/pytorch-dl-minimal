## {{ cookiecutter.project_name }}

Directory structure:
```
├── {{ cookiecutter.project_name }}
│   ├── main.py  # starting point
│   ├── models
│   │   └── __init__.py  # contains models
│   ├── solver.py  # contains code for training using {{ cookiecutter.project_name }}
│   └── utils
│       ├── __init__.py  # contains additional code used
├── Makefile
├── pretrained
│   └── lenet5.pth.tar
├── readme.md
└── requirements.txt
```
## Setup

Make sure the requirements for [pytorch](https://pytorch.org/get-started/locally/) are satisfied. To create the env run:

```
$ make venv
```

## Run

Make necessary changes in the `Makefile` rules and execute them.
