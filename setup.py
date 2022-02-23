import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open('VERSION') as f:
    version = f.read()

setuptools.setup(
    name='modeleval',
    version=version,
    description='Evaluation of mlflow binary classifiers',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/hector-pinheiro/classifier-evaluation',
    packages=setuptools.find_packages(exclude=['test']),
    install_requires=required,
)
