# 概要

Pythonのプロジェクトを作る際のベースとなるプロジェクトです。

このリポジトリをcloneし、.gitを削除してからgit initしなおし、新しいプロジェクトして成長させていって
下さい。

大事な要素としては

1. Pipenvを使ったパッケージとPythonバージョンの管理
2. pytestを使ったテストの実行
3. pre-commitを使ったコードのフォーマットと静的解析

までが含まれています。ここまではチームでの共通のツールとして使っていきましょう。

# セットアップ

## pipenvのインストール

すでにpipenvが入っている場合は飛ばして下さい。

pipenvはUbuntuやDebian系のLinuxディストリビューションでは、以下のコマンドで
インストールできるはずです。

```bash
$ sudo apt update
$ sudo apt upgrade
$ sudo apt install python3-pip
$ pip3 install --user pipenv
```

## プロジェクトのセットアップ

gitリポジトリのクローン。

```bash
$ git clone git@github.com:stc-zao-developer/python-project-base.git
```

プロジェクトディレクトリに移動して、pipenvで依存関係をインストール。

```bash
$ cd python-project-template
$ pipenv install --dev
$ pipenv install
```

pre-commitのセットアップ

```bash
$ pipenv run pre-commit install
```

pre-commitをセットアップすることにより、gitのcommit時に自動でコードの静的解析が実行されるように
なり、静的解析で検出された問題はコミットできなくなります。
これにより、コードの品質を保つことができます。

# 開発

ソースコードはsrcディレクトリ以下に配置されます。

```bash
$ pipenv run start
```

を実行することにより、main.pyが実行されます。

## テストの実行

テストはtestsディレクトリ以下に配置されます。
pytestを使って実行します。

```bash
$ pipenv run pytest
```

また、
```bash
$ pipenv run watch
```

を実行することにより、コードの変更を監視し、変更があった場合に自動でテストを実行します。
とてもおすすめです。

## コードのフォーマット

コードのテストは、pre-commitで自動的に行われまが、手動で実行することもできます。

```bash
$ pipenv run check
```

pre-commitや、checkコマンドで指摘された問題を自動で修正するには

```bash
$ pipenv run fix
```

を実行します。
