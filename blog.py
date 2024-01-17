'''

# はじめに
本記事では、YOLOv8モデルを使用して推論を行う際の前処理と後処理の手順について詳しく解説します。
特に、ポーズ推定タスクにおける顔認識を例に、一般的な処理方法を紹介します。後処理のkps（キーポイント）に関する部分以外は他のタスクにも応用できます。

コードは[github](https://github.com/tamoharu/YOLOv8-example.git)で公開しています。

# 基本的な流れ
1. 前処理
2. 推論
3. 後処理

# モデルの準備
ultralyticsライブラリでは、pytorch形式のモデルを様々なフォーマットにエクスポートする機能があります。
以下のコードを実行することで、モデルをエクスポートします。
```python
from ultralytics import YOLO
model = YOLO('../models/yolov8n-face.pt')
model.export(format='onnx', imgsz=640, opset=12, dynamic=True) # dynamic
# model.export(format='onnx', imgsz=640, opset=12, dynamic=False) # static
```
`dynamic=True`オプションを入れることで、モデルの入出力形式が可変になります。
これによって、複数枚の画像を一度に推論するバッチ処理が可能になります。

モデルの入出力形式は以下の通りです。
```
input: (n, 3, 640, 640)
output: (n, 20, 8400)
```

# 型定義
まず、処理に使う値の型を定義しておきます。
考えるのがめんどくさかったのでそこまで厳密にはしてないです。
```python
Session = onnxruntime.InferenceSession
Image = numpy.ndarray[Any, Any]
Prediction = numpy.ndarray[Any, Any]
MetaData = dict[str, list[int]]
Bbox = numpy.ndarray[Any, Any]
Kps = numpy.ndarray[Any, Any]
Score = float
Face = namedtuple('Face',
[
	'bbox',
	'kps',
	'score',
])
```

# 前処理
推論に使用する画像をモデルが要求する入力形状に合わせて変換する必要があります。このモデルは入力として`(n, 3, 640, 640)`の形状を期待していて、これは`(バッチ数, チャネル数, 高さ, 幅)`です。以下、`(B, C, H, W)`のように表記します。
OpenCVで読み込まれた画像は通常`(H, W, C)`形式であり、チャネルがRGBの順になっています。この画像をモデルの入力形状に適した形に変換するのが前処理の目的です。
具体的な手順は以下の通りです。

**1. リサイズ**
元のアスペクト比を維持しつつパディングを追加するレターボックス加工を行います。
また、後処理で画像のサイズを元に戻すために、リサイズに使った値をmetadataとして保存しておきます。

**2. 正規化**
画像の各ピクセルは0~255の整数で表されます。
これをニューラルネットワークの入力に合うように0~1のスケールに変換します。

**3. チャネル変換**
OpenCVでは画像はBGRの順番で読み込まれますが、ニューラルネットワークに渡す際にはRGBの順番が必要なので、チャネルの順序をBGRからRGBに変更します。

**4. 行列の転置**
画像の形式を`(H, W, C)`から`(C, H, W)`に変換します。

以下はこれらを実装したコードです。
```python
def preprocess_images(images: List[Image], input_size=(640, 640)) -> tuple[numpy.ndarray[Any, Any], MetaData]:
	preprocessed = []
	meta_data = {'ratios': [], 'dws': [], 'dhs': []}
	for image in images:
		# resize
		shape = image.shape[:2]
		ratio = min(input_size[0] / shape[0], input_size[1] / shape[1])
		new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
		dw, dh = (input_size[1] - new_unpad[0]) / 2, (input_size[0] - new_unpad[1]) / 2
		image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
		top, bottom = round(dh - 0.1), round(dh + 0.1)
		left, right = round(dw - 0.1), round(dw + 0.1)
		image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
		
		# normalize
		image = image.astype(numpy.float32) / 255.0
		
		# RGB -> BGR
		image = image[..., ::-1]
		
		# HWC -> CHW
		image = image.transpose((2, 0, 1))
		preprocessed.append(image)
		meta_data['ratios'].append(ratio)
		meta_data['dws'].append(dw)
		meta_data['dhs'].append(dh)
	return numpy.ascontiguousarray(numpy.array(preprocessed)), meta_data
```

# 推論
前処理で準備された画像をモデルに渡し、推論を行います。
なお、並列処理を行うことを想定してセマフォを使用しています。
```python
def predict(model: Session, images: Image) -> Prediction:
	with threading.Semaphore():
		start_time = time.time()
		predictions = model.run(None, {model.get_inputs()[0].name: images})[0]
		print(f'inference time: {time.time() - start_time}')
	return predictions
```

# 後処理
推論結果を加工して、利用しやすい形にします。
処理の解説の前に、モデルから出力される情報がどのようなものかを確認しましょう。

### 返却される情報
本モデルの場合、返却される情報は以下の通りです。
なお、outputの形状はモデルによって異なります。

**boundingbox(4次元)**
検出された顔の領域を長方形で囲ったもの。
長方形の中心座標、幅、高さが`(x_center, y_center, width, height)`の形式で表される。

**score(1次元)**
各検出がどの程度信頼できるかを表すスコア。
0~1の範囲で表される。

**keypoints(15次元)**
ポーズ推定におけるキーポイントの座標。
目(2点)、鼻(1点)、口(2点)の座標と各点の信頼度が含まれている。
`(x1, y1, score1, x2, y2, score2, ...)`の形式で表される。

outputの形状は`(n, 20, 8400)`です。
これは上記に示した情報(20次元)が8400個格納されているという意味です。
モデルは常に8400個分の検知を出力しています。

これらを踏まえて、後処理を実装していきます。

### 実装
以下の手順で実装します。

**1. 行列の転置**
得られた結果を扱いやすい形にするため、`(n, 8400, 20)`の形に変形します。
これにより、predictionは`[bbox, score, kps]`の形になります。
具体的にpredictionsは以下のようになります。
```
[[x_center, y_center, width, height, score, x1, y1, score1, x2, y2, score2, ...], # 20個の要素を持つ
[x_center, y_center, width, height, score, x1, y1, score1, x2, y2, score2, ...],
...
# 8400個続く
```
ここから、predictionsの各要素に対して処理を行います。

**2. bboxの座標変換・リサイズ**
bboxは`(x_center, y_center, width, height)`の形になっていますが、これを扱いやすいように、bboxの左上の座標、右下の座標の形式`(x_min, y_min, x_max, y_max)`に変換します。
また、bboxのスケールを元の画像に戻します。

**3. kpsの座標抽出・リサイズ**
kpsは`(x1, y1, score1, x2, y2, score2, ...)`の形になっていますが、これを扱いやすいように、`(x1, y1), (x2, y2), ...`の形に変換します。
また、kpsのスケールを元の画像に戻します。

**4. scoreの値でフィルタリング**
モデルは8400個もの検出結果を出力しますが、そのほとんどは信頼度が非常に低いものです。
そのため、一定の閾値を設けて、信頼度が低いものを除去します。

**5. NMSによるbboxの絞り込み**
通常、モデルは一つの対象に対して複数の検出結果を出力してしまいます。
これをNMS(Non-Maximum Suppression)という手法を用いて絞り込みます。
これにより、多くの場合で同じ対象に対する検出結果を一つに絞り込むことができます。

**6. リスト化**
最後に、各要素をリストに変換し、扱いやすい形にします。

```python
def postprocess_predictions(predictions: Prediction, meta_data: MetaData, score_threshold=0.25, iou_threshold=0.4) -> List[Face]:
	# (n, 20, 8400) -> (n, 8400, 20)
	predictions = numpy.transpose(predictions, (0, 2, 1))
	predictions = numpy.ascontiguousarray(predictions)
	
	# create batch faces
	batch_faces = []
	for i, pred in enumerate(predictions):
		bbox, score, kps = numpy.split(pred, [4, 5], axis=1)
		ratio, dw, dh = meta_data['ratios'][i], meta_data['dws'][i], meta_data['dhs'][i]
		
		# (x_center, y_center, width, height) -> (x_min, y_min, x_max, y_max)
		# restore to original size
		new_ratio = 1/ratio	
		x_center, y_center, width, height = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
		x_min = (x_center - (width / 2) - dw) * new_ratio
		y_min = (y_center - (height / 2) - dh) * new_ratio
		x_max = (x_center + (width / 2) - dw) * new_ratio
		y_max = (y_center + (height / 2) - dh) * new_ratio
		bbox = numpy.stack((x_min, y_min, x_max, y_max), axis=1)
		
		# (x, y, score) -> (x, y)
		# restore to original size
		for i in range(kps.shape[1] // 3):
			kps[:, i * 3] = (kps[:, i * 3] - dw) * new_ratio
			kps[:, i * 3 + 1] = (kps[:, i * 3 + 1] - dh) * new_ratio
		
		# filter
		indices_above_threshold = numpy.where(score > score_threshold)[0]
		bbox = bbox[indices_above_threshold]
		score = score[indices_above_threshold]
		kps = kps[indices_above_threshold]
		
		# nms
		nms_indices = cv2.dnn.NMSBoxes(bbox.tolist(), score.ravel().tolist(), score_threshold, iou_threshold)
		bbox = bbox[nms_indices]
		score = score[nms_indices]
		kps = kps[nms_indices]
		
		# convert to list
		bbox_list = []
		for box in bbox:
			bbox_list.append(numpy.array(
			[
				box[0],
				box[1],
				box[2],
				box[3],
			]))
		score_list = score.ravel().tolist()
		kps_list = []
		for keypoints in kps:
			kps_xy = []
			for i in range(0, len(keypoints), 3):
				kps_xy.append([keypoints[i], keypoints[i+1]])
			kps_list.append(numpy.array(kps_xy))

		batch_faces.append([Face(bbox=bbox, kps=kps, score=score) for bbox, kps, score in zip(bbox_list, kps_list, score_list)])
	return batch_faces
```

### 結果の描画
うまくいっているかどうかを確認するために、結果を描画してみましょう。
以下のコードで確認できます。
```
def draw_results(images: List[Image], batch_faces: List[Face]) -> None:
	for i, (image, faces) in enumerate(zip(images, batch_faces)):
		bbox_list = []
		kps_list = []
		for face in faces:
			bbox_list.append(face.bbox)
			kps_list.append(face.kps)
		for bbox, keypoints in zip(bbox_list, kps_list):
			start_point = (int(bbox[0]), int(bbox[1]))
			end_point = (int(bbox[2]), int(bbox[3]))
			image = cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
			for kp in keypoints:
				x, y = int(kp[0]), int(kp[1])
				image = cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
		cv2.imshow(f'image{i}', image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
```

# まとめ
以上で、YOLOv8モデルを使用して推論を行う際の前処理と後処理の手順について解説しました。
YOLOv8モデルを使用することで、顔認識をはじめとする様々なタスクを簡単に実装することができます。
ぜひ、ご活用ください。
'''