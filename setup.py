from setuptools import find_packages, setup
import warnings

DEPENDENCY_PACKAGE_NAMES = [
    "matplotlib",
    "torch",
    "tqdm",
    "numpy",
    "opencv-python",
    "scipy",
    "chumpy",
    "trimesh",
    "pyvista",
    "deprecation",
    "open3d",
]


def check_dependencies():
    missing_dependencies = []
    for package_name in DEPENDENCY_PACKAGE_NAMES:
        try:
            __import__(package_name)
        except ImportError:
            missing_dependencies.append(package_name)

    if missing_dependencies:
        warnings.warn('Missing dependencies: {}. We recommend you follow '
                      'the installation instructions at '
                      'https://github.com/Junble19768/manotorch#installation'.format(missing_dependencies))


# with open("README.md", "r") as fh:
#     long_description = fh.read()

# check_dependencies()

setup(
    name="manotorch",
    version="0.0.3",
    author="Baosong Jiang",
    author_email="junble@mail.dlut.edu.cn",
    packages=find_packages(exclude=('tests',)),
    python_requires=">=3.7.0",
    description="MANO pyTORCH",
    # long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Junble19768/manotorch",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE",
        "Operating System :: OS Independent",
    ],
    install_requires=DEPENDENCY_PACKAGE_NAMES,
)
