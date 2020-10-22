import setuptools

with open('README.md', 'r') as ld:
    long_description = ld.read()

setuptools.setup(
    name="StockPredictorLSTM-Vyzrala",
    version="0.1.1",
    description="Thesis project",
    long_description=long_description,
    author="Marcin Hebdzyński",
    author_email="hebdzynski.m@gmail.com",
    url="https://github.com/Vyzrala/StockPredictorLSTM",
    packages=setuptools.find_packages(include=["StockPredictorLSTM", "StockPredictorLSTM.*"], exclude=["tests"], where="StockPredictorLSTM"),
    package_dir = {":":"StockPredictorLSTM"},
    python_requires=">=3.7",
)