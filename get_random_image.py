import pathlib
import urllib.request

req_width = 1920
req_height = 1080
extension = '.jpg'
dir_path = pathlib.Path('sample-inputs')
url = f'https://picsum.photos/{req_width}/{req_height}{extension}'
for i in range(100):
    path = dir_path / f'lorem-picsum-{i}{extension}'
    if not path.exists():
        print(f'Requesting image from {url}')
        urllib.request.urlretrieve(url, path)
