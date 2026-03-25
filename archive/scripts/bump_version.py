from pathlib import Path
import re
import sys


VERSION_PATTERN = re.compile(r"^version = '(\d+)\.(\d+)\.(\d+)'$", re.MULTILINE)


def bump_patch_version(pyproject_filename):
    '''Bump the patch version in pyproject.toml.'''
    path = Path(pyproject_filename)
    text = path.read_text()
    match = VERSION_PATTERN.search(text)
    if match is None:
        raise ValueError('Could not find version in pyproject.toml')

    major, minor, patch = (int(value) for value in match.groups())
    bumped_version = f"{major}.{minor}.{patch + 1}"
    updated_text = VERSION_PATTERN.sub(
        f"version = '{bumped_version}'",
        text,
        count=1,
    )
    path.write_text(updated_text)
    return bumped_version


def main():
    filename = 'pyproject.toml'
    if len(sys.argv) > 1:
        filename = sys.argv[1]

    bumped_version = bump_patch_version(filename)
    print(f'Bumped version to {bumped_version}')


if __name__ == '__main__':
    main()
