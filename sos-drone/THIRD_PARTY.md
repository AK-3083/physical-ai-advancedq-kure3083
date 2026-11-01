# 使用環境・ライセンス

マーカー・負例・学習画像・地形・テクスチャ・樹木は本プロジェクトで生成します。
学習済みモデルも生成画像で学習しています。人物写真、実山の地図、外部モデルの重みは使用しません。

| 使用元 | 用途・ライセンス |
|---|---|
| [sjtu_drone](https://github.com/NovoG93/sjtu_drone) | 機体、Gazeboプラグイン、位置制御、カメラ、ROSトピック。GPL-3.0 |
| [ROS 2 Humble](https://docs.ros.org/en/humble/) | rclpy、標準メッセージ、cv_bridge、Gazebo ROS。各パッケージのライセンス |
| [Gazebo Classic 11](https://classic.gazebosim.org/) | 物理・描画・カメラ。Apache-2.0。サポート終了済み |
| [ROSデスクトップVNC](https://github.com/Tiryoh/docker-ros2-desktop-vnc) | DockerのUbuntu・ROS・VNC環境。内包ソフトウェアのライセンスも適用 |
| [PyTorch CPU 2.5.1](https://pytorch.org/) | CNN学習。BSD-style |
| [ONNX 1.16.2](https://onnx.ai/) | モデル保存。Apache-2.0 |
| [OpenCV](https://opencv.org/) | 画像生成、推論、動画処理。コンテナの4.5.4はApache-2.0 |
| [NumPy](https://numpy.org/) | 数値計算。BSD-3-Clause |
| [FFmpeg](https://ffmpeg.org/)・[libx264](https://www.videolan.org/developers/x264.html) | 画面録画、H.264変換。GPL等、ビルド構成に応じた条件 |
| [Ruff](https://docs.astral.sh/ruff/) | Lint用ツール。MIT |

