from setuptools import find_packages, setup

package_name = 'BeBOP'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jstyrud',
    maintainer_email='45998292+jstyrud@users.noreply.github.com',
    description='BeBOP',
    license='Apache License 2.0',
    tests_require=['pytest'],
)