# 実行まで

1. 仮想環境を構築
```
python -m venv venv
source venv/bin/activate
```

2. パッケージをインストール
```
pip install -r requirements.txt
```

3. 画像をセット
`images`ディレクトリに検知したい画像をセット
`run.py`のimage_pathsを編集

4. 実行
```
python run.py
```


#### References

* [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

* [https://github.com/derronqi/yolov8-face.git](https://github.com/derronqi/yolov8-face.git)