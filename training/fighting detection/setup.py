from setuptools import setup, find_packages

setup(
    name="Trial1",
    version="0.1.0",
    author="Human Action Recognition team",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "albumentations",
        "ipython",
        "moviepy",
        "opencv-python",
        "pandas",
        "protobuf",
        "pytube",
        "pytube3",
    ],
    zip_safe=False,
)
