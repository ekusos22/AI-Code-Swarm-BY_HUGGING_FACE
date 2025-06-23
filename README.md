# 組織的AIコーディングシステム (Hugging Face版)

このシステムは、Hugging Face HubでホストされているオープンソースLLM（大規模言語モデル）を利用して、指定されたアプリケーションを自動で開発するPythonスクリプトです。

## 登場するAIエージェント

- **👑 President AI:** プロジェクトの最高責任者。
- **📋 Project Manager AI:** 開発タスクを計画し、`Project/README.md`に記録します。
- **👷 Engineer AI (x2):** `Project/README.md`のタスクに従い、コーディングを行います。

## 使い方

### 1. 準備

1.  **Hugging Face アカウントとアクセストークン**
    - [Hugging Face](https://huggingface.co/) のアカウントをお持ちでない場合は作成してください。
    - [設定 > Access Tokens](https://huggingface.co/settings/tokens) から新しいアクセストークン（Read権限でOK）を生成し、コピーします。

2.  **ライブラリのインストール**
    ```bash
    pip install huggingface_hub python-dotenv
    ```

3.  **.envファイルの設定**
    このプロジェクトと同じフォルダに `.env` ファイルを新規作成し、コピーしたHugging Faceアクセストークンを以下のように記述します。
    ```
    HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxx"
    ```

4.  **`request.txt` ファイルに開発内容を記述**
    `request.txt` ファイルに、開発してほしいアプリケーションの概要を記述してください。

    **`request.txt` の記述例:**
    ```
    コマンドラインで動作する簡単なTODOリストアプリケーションを作成してください。
    機能は以下の通りです。
    - タスクを追加する機能 (add)
    - タスクを一覧表示する機能 (list)
    タスクは `tasks.json` ファイルに保存すること。
    ```

### 2. 実行

以下のコマンドでスクリプトを実行します。

```bash
python main.py