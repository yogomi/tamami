# 概要

Pythonのプロジェクトを作る際のベースとなるプロジェクトです。

このリポジトリをcloneし、.gitを削除してからgit initしなおし、新しいプロジェクトして成長させていって
下さい。

大事な要素としては

1. uvを使ったパッケージとPythonバージョンの管理
2. pytestを使ったテストの実行
3. pre-commitを使ったコードのフォーマットと静的解析

までが含まれています。ここまではチームでの共通のツールとして使っていきましょう。

# セットアップ

## 依存ライブラリのインストール

PyAudioを使用するため、システムにportaudioライブラリが必要です。

**macOS:**
```bash
$ brew install portaudio
```

**Ubuntu/Debian:**
```bash
$ sudo apt-get install portaudio19-dev
```

## uvのインストール

すでにuvが入っている場合は飛ばして下さい。

uvは以下のコマンドでインストールできます。

```bash
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```

macOSではHomebrewでもインストールできます。

```bash
$ brew install uv
```

## プロジェクトのセットアップ

gitリポジトリのクローン。

```bash
$ git clone git@github.com:stc-zao-developer/python-project-base.git
```

プロジェクトディレクトリに移動して、uvで依存関係をインストール
（dev依存を含めて `.venv` に同期されます）。

```bash
$ cd tamami
$ uv sync
```

なお、`uv.lock` はCUDA/CPUなど環境ごとに依存が変わるためコミットせず、
各環境で生成する運用としています（`.gitignore` 済み）。

### CPU専用環境（CUDAなし）での設定

GPUがない環境やCUDAがインストールされていない環境では、CPU専用のPyTorchを使用できます。
`uv sync` の後に、以下のコマンドでCPU専用のPyTorchを上書きインストールしてください：

```bash
$ uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

（`uv sync` を実行し直すと元に戻るため、その場合は再度実行してください）

pre-commitのセットアップ

```bash
$ uv run pre-commit install
```

pre-commitをセットアップすることにより、gitのcommit時に自動でコードの静的解析が実行されるように
なり、静的解析で検出された問題はコミットできなくなります。
これにより、コードの品質を保つことができます。

# 開発

ソースコードはsrcディレクトリ以下に配置されます。

```bash
$ uv run python -m src.main
```

を実行することにより、main.pyが実行されます。

ストリーミング翻訳サーバーを起動する場合は以下を実行します。

```bash
$ uv run python -m src.server
```

## テストの実行

テストはtestsディレクトリ以下に配置されます。
pytestを使って実行します。

```bash
$ uv run pytest -s
```

また、
```bash
$ uv run ptw --config pytest.ini --runner 'pytest --testmon -s'
```

を実行することにより、コードの変更を監視し、変更があった場合に自動でテストを実行します。
とてもおすすめです。

## コードのフォーマット

コードのテストは、pre-commitで自動的に行われまが、手動で実行することもできます。

```bash
$ uv run pre-commit run --all-files
```

pre-commitや、checkコマンドで指摘された問題を自動で修正するには

```bash
$ uv run ruff check . --fix && uv run black .
```

を実行します。
