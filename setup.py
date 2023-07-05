import setuptools

setuptools.setup(
    include_package_data=True,
    name="mle_training",
    version="0.1.0",
    author="Vislavath Praveen",
    package_dir={
        "": "src",
    },
    packages=setuptools.find_packages(where="src"),
    description="new mle package ",
    long_description=open("README.md").read(),
    install_requires=[],
)
