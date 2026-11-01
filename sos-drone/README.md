# 山中のマーカー探索ドローン

ROS 2 Humble・Gazebo Classic 11で、山道を飛行し、人工マーカーをCNNで検出するPoCです。
継続検出すると停止して近傍3視点を撮影し、探索終了後に麓へ帰還・着陸します。
実際の人物探索・救助や未知経路の探索は対象外です。位置はGazeboのローカルXYZです。

## 必要環境

- WindowsのDocker Desktop（Linuxコンテナ）とブラウザ
- 初回ビルド用ネットワーク、数GB以上の空きディスク、十分なDockerメモリ

以下を上から順番に実行します。Ubuntuの各コードブロックは、新しい端末でもコピーして実行できます。
同じROS domainで他の飛行・録画・評価を同時に実行しないでください。

## 1. コンテナを起動する（Windows PowerShell）

このREADMEがある `sos-drone` フォルダで実行します。
リポジトリのルートにいる場合は、先に `Set-Location .\sos-drone` を実行してください。
既存コンテナがあれば起動し、なければビルド・作成します。
既存コンテナは、このフォルダを `/workspace/sos` に共有した本PoCの環境を使ってください。

```powershell
$existing = docker container ls -a --filter 'name=^/pai-sos-mountain$' --format '{{.Names}}'
if ($LASTEXITCODE -ne 0) { throw 'Docker Desktopを起動してください。' }
if ($existing -eq 'pai-sos-mountain') {
    docker start pai-sos-mountain
} else {
    docker compose build
    if ($LASTEXITCODE -ne 0) { throw 'ビルドに失敗しました。ログを確認してください。' }
    docker compose up -d --no-build
}
if ($LASTEXITCODE -ne 0) { throw 'コンテナの起動に失敗しました。' }
for ($attempt = 0; $attempt -lt 40; $attempt++) {
    curl.exe --fail --silent --output NUL --max-time 2 http://127.0.0.1:6082/
    if ($LASTEXITCODE -eq 0) { break }
    Start-Sleep -Seconds 2
}
if ($LASTEXITCODE -ne 0) { throw 'デスクトップに接続できません。コンテナのログを確認してください。' }
```

ブラウザで <http://127.0.0.1:6082/> を開き、`Connect` を押します。
以降は表示されたUbuntuデスクトップの端末で実行します。

## 2. モデルを準備する（Ubuntu・端末A）

モデルが揃っていれば既存モデルを使い、なければ学習します。完了後に次へ進みます。

```bash
set +e
cd /workspace/sos
source /opt/ros/humble/setup.bash
source /opt/drone/install/setup.bash
export DISPLAY=:1 XAUTHORITY=/home/ubuntu/.Xauthority
export LIBGL_ALWAYS_SOFTWARE=1 ROS_DOMAIN_ID=78
if [ ! -f models/color/marker.onnx ] || [ ! -f models/color/metrics.json ]; then
    /opt/sos/train/bin/python -m scripts.train --output models/color
else
    echo "学習済みモデルを使用します。"
fi
```

## 3. シーンを生成する（端末A）

マーカー2個のシーンを生成します。
生成に成功してから手順4へ進みます。

```bash
set +e
cd /workspace/sos
source /opt/ros/humble/setup.bash
source /opt/drone/install/setup.bash
export DISPLAY=:1 XAUTHORITY=/home/ubuntu/.Xauthority
export LIBGL_ALWAYS_SOFTWARE=1 ROS_DOMAIN_ID=78
RUN_NAME="color-manual"
RUN_DIR="runs/$RUN_NAME"
SCENE_DIR="${RUN_DIR}-scene"
if pgrep -x gzserver >/dev/null || pgrep -x gzclient >/dev/null; then
    printf 'GazeboとGUIを終了してから手順3を実行してください。\n' >&2
    false
else
    rm -rf -- \
        /workspace/sos/runs/color-manual \
        /workspace/sos/runs/color-manual-scene \
        /workspace/sos/deliverables/color-manual &&
    /opt/sos/train/bin/python -m scripts.mountain_scene --directory "$SCENE_DIR" --seed 503 --count 2
fi
```

## 4. Gazeboを起動する（端末A）

端末Aは実行したままにします。機体が生成されてから手順5へ進みます。

```bash
set +e
cd /workspace/sos
source /opt/ros/humble/setup.bash
source /opt/drone/install/setup.bash
export DISPLAY=:1 XAUTHORITY=/home/ubuntu/.Xauthority
export LIBGL_ALWAYS_SOFTWARE=1 ROS_DOMAIN_ID=78
RUN_NAME="color-manual"
SCENE_DIR="runs/${RUN_NAME}-scene"
export SOS_WORLD="/workspace/sos/$SCENE_DIR/search.world"
ros2 launch scripts/sim.launch.py
```

## 5. Gazebo GUIを表示する（新しい端末B）

草地・樹木・ドローンが表示されます。端末Bも実行したままにします。

```bash
set +e
cd /workspace/sos
source /opt/ros/humble/setup.bash
source /opt/drone/install/setup.bash
export DISPLAY=:1 XAUTHORITY=/home/ubuntu/.Xauthority
export LIBGL_ALWAYS_SOFTWARE=1 ROS_DOMAIN_ID=78
gzclient --verbose
```

## 6. 枠付き判定画像を表示する（新しい端末C）

Gazeboとは別の画像ウィンドウを開きます。枠と数値が読めるよう、ウィンドウの端をドラッグして広げます。飛行開始前は画像待ちになります。
Gazebo内のカメラ画像そのものには枠を描きません。

```bash
set +e
cd /workspace/sos
source /opt/ros/humble/setup.bash
source /opt/drone/install/setup.bash
export DISPLAY=:1 XAUTHORITY=/home/ubuntu/.Xauthority
export LIBGL_ALWAYS_SOFTWARE=1 ROS_DOMAIN_ID=78
ros2 run rqt_image_view rqt_image_view --on-top /sos/preview/image
```

- 黄色枠：色から抽出した候補
- 緑色枠：CNNスコアが閾値以上の候補（この枠だけで地点確定とはしません）
- 枠付近の数値：CNNスコア。校正済みの正解確率ではありません。
- 下部：判定画像の撮影時刻と凡例

最新の判定画像を表示するため、探索中は約0.4秒間隔、近傍撮影時は撮影ごとに更新します。
離陸・視点移動・帰還など、判定していない区間では表示の更新が止まります。

## 7. 自動飛行する（新しい端末D）

枠表示、継続2画像での検出後の停止・近傍撮影、探索再開、帰還・着陸を確認します。
飛行が終了してプロンプトに戻ってから手順8へ進みます。

```bash
set +e
cd /workspace/sos
source /opt/ros/humble/setup.bash
source /opt/drone/install/setup.bash
export DISPLAY=:1 XAUTHORITY=/home/ubuntu/.Xauthority
export LIBGL_ALWAYS_SOFTWARE=1 ROS_DOMAIN_ID=78
RUN_NAME="color-manual"
RUN_DIR="runs/$RUN_NAME"
SCENE_DIR="${RUN_DIR}-scene"
/opt/sos/train/bin/python -m sos.mission --terrain "$SCENE_DIR/terrain.json" --output "$RUN_DIR" --publish-preview
```

## 8. 結果を確認して終了する（端末D）

`complete` と発見地点2件を確認します。未完了・件数不一致はエラーになります。

```bash
set +e
cd /workspace/sos
RUN_NAME="color-manual"
RUN_DIR="runs/$RUN_NAME"
/opt/sos/train/bin/python - "$RUN_DIR/report.json" <<'PY'
import json
from pathlib import Path
import sys

report = json.loads(Path(sys.argv[1]).read_text())
print("status:", report["status"])
print("sites:", len(report["sites"]))
for site in report["sites"]:
    print(f'XYZ: ({site["x"]:.2f}, {site["y"]:.2f}, {site["z"]:.2f})')
if report["status"] != "complete" or len(report["sites"]) != 2:
    raise SystemExit("飛行未完了または発見地点数が一致しません。")
PY
```

結果はWindows側の `runs/color-manual` でも確認できます。

| ファイル | 内容 |
|---|---|
| `report.json`・`sites.csv` | 完了状態、推定XYZ、統合後の発見地点 |
| `scan_*.png` | 近傍撮影時の枠付き静止画像 |
| `evidence_*.png` | 枠を描いていない証拠画像 |
| `flight-metrics.json` | 撮影数、接触有無、飛行指標 |

端末C、端末B、端末Aで順番にCtrl+Cを押し、画像ビューア、Gazebo GUI、Gazeboを終了します。
再実行は手順3から行います。専用の録画・動画編集は不要です。

[使用環境・参考元・ライセンス](THIRD_PARTY.md)
