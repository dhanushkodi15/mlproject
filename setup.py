from setuptools import setup, find_packages

def get_requires(file_path:str)->list[str]:
    """Reads the requirements file and returns a list of dependencies."""
    with open(file_path, 'r') as file:
        requires = file.readlines()
    return [req.replace("\n","") for req in requires if req != '-e .']

setup(
    name = "ML Project",
    version = "1.0",
    author="Dhanush kodi",
    author_email="gdevkodi15@gmail.com",
    packages = find_packages(),
    install_requires=get_requires('requirements.txt')
)
