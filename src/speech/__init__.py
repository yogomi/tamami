"""ストリーミング音声認識モジュール.

StreamingRecognizer（抽象基底、streaming.py）とその実装（fake.py / nemotron.py）を
提供する。実装クラスはimport時の副作用（NeMo等の重い依存の読み込み）を避けるため
再エクスポートせず、利用側が各モジュールから直接importする。
"""
