'''
https://svaderia.github.io/articles/downloading-and-unzipping-a-zipfile/
'''
from urllib.request import urlopen
from zipfile import ZipFile
import os


def download_movielens(url, file_path='./data/movielens'):
    r = urlopen(url)
    zf_name = url.split('/')[-1] # 압축 파일 이름

    # 데이터셋 압축 파일 다운로드
    if not os.path.exists(f'{file_path}/{zf_name}'):
        print(f'파일 다운로드: {url}')
        with open(f'{file_path}/{zf_name}', 'wb') as f:
            f.write(r.read())

    # 압축 풀기
    zf = ZipFile(f'{file_path}/{zf_name}')
    zf.extractall(path=f'{file_path}')
    zf.close()


if __name__ == '__main__':
    URL = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
    download_movielens(URL)

