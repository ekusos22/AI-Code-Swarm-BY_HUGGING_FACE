import os
import re
import time
import shutil
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from collections import defaultdict

# --- 設定 ---
# .envファイルから環境変数を読み込む
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Hugging Face InferenceClientの初期化
try:
    if not HF_TOKEN:
        raise ValueError("環境変数 `HF_TOKEN` が設定されていません。")
    client = InferenceClient(token=HF_TOKEN)
except Exception as e:
    print(f"Hugging Faceクライアントの初期化に失敗しました: {e}")
    client = None

# 使用するAIモデル (Hugging Face HubのモデルID)
MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# プロジェクト設定
PROJECT_DIR = "Project"
REQUEST_FILE = "request.txt"

# --- ヘルパー関数 ---

def clean_project_dir():
    """Projectディレクトリの中身を削除する。"""
    if not os.path.exists(PROJECT_DIR):
        return
    for filename in os.listdir(PROJECT_DIR):
        file_path = os.path.join(PROJECT_DIR, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"ファイル/ディレクトリの削除中にエラーが発生しました: {e}")

def create_project_dir():
    """プロジェクト用のディレクトリを作成する。"""
    if not os.path.exists(PROJECT_DIR):
        os.makedirs(PROJECT_DIR)

def read_file(filepath):
    """ファイルを読み込む。存在しない場合は空文字列を返す。"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

def write_file(filepath, content):
    """ファイルに書き込む。ディレクトリが存在しない場合は作成する。"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

# --- AIエージェントの定義 ---

def ai_call(system_prompt, user_prompt, max_retries=3):
    """Hugging Face Inference APIを呼び出す共通関数。"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    for attempt in range(max_retries):
        try:
            print(f"🧠 AI ({MODEL}) is thinking...")
            response = client.chat_completion(
                messages=messages, model=MODEL, temperature=0.1, max_tokens=4096, stream=False,
            )
            print("✅ AI response received.")
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ API呼び出しエラー (試行 {attempt + 1}/{max_retries}): {e}")
            time.sleep(10)
    print("❌ AI呼び出しに失敗しました。")
    return None

def president_ai(user_request):
    """President AI: ユーザーの要求からプロジェクトの基本方針を決定する。"""
    system_prompt = "あなたは企業のPresidentです。ユーザーの要求を元に、開発プロジェクトの基本方針と概要を決定し、Project Managerに指示を出してください。出力は簡潔な指示形式で、Markdownで記述してください。挨拶や署名などの余計なテキストは一切含めないでください。"
    user_prompt = f"ユーザーからの開発要求:\n---\n{user_request}\n---\n上記の要求を元に、Project Managerへの指示を作成してください。"
    print("\n===== 👑 President AI's Turn =====")
    instruction = ai_call(system_prompt, user_prompt)
    if instruction:
        print("▶️ PresidentからPMへの指示:\n", instruction)
    return instruction

def project_manager_ai(president_instruction):
    """Project Manager AI: Presidentの指示を具体的なタスクリストに分解し、README.mdを作成する。"""
    system_prompt = (
        "あなたは優秀なProject Managerです。"
        "Presidentの指示を元に、具体的な開発タスクリストを`README.md`に書き込むためのコンテンツを作成してください。"
        "重要: 全てのタスクに、対象ファイル名を必ずバッククォート(`)で囲んで明記し、未完了を示す `[ ]` を付けてください。"
        "悪い例: - [ ] メインウィンドウを作成する。\n"
        "良い例: - [ ] `main.py`にメインウィンドウを作成する。\n"
        "出力は`README.md`に書き込むMarkdownタスクリストのみとしてください。挨拶や説明などの余計なテキストは絶対に含めないでください。"
    )
    user_prompt = f"Presidentからの指示:\n---\n{president_instruction}\n---\n上記の指示を、全てのタスクにファイル名を含む具体的なタスクリストに変換してください。"
    print("\n===== 📋 Project Manager AI's Turn =====")
    new_readme_content = ai_call(system_prompt, user_prompt)
    
    if new_readme_content:
        # AIが余計なMarkdownブロックを付けてしまう場合があるので除去する
        new_readme_content = re.sub(r'^```(markdown)?\n', '', new_readme_content, flags=re.IGNORECASE)
        new_readme_content = re.sub(r'\n```$', '', new_readme_content)
        write_file(os.path.join(PROJECT_DIR, "README.md"), new_readme_content)
        print("✅ README.md を作成/更新しました。")
    else:
        print("❌ PMがREADMEの生成に失敗しました。")
    return new_readme_content is not None

def engineer_ai(task, engineer_id, fallback_filename=None):
    """Engineer AI: タスクを実行してコードを生成する。"""
    system_prompt = (
        "あなたは優秀なPython Engineerです。指示に従って、コードを生成・修正してください。"
        "あなたの仕事は、指定されたファイルに書き込むための完全なコードを生成することです。"
        "重要: 出力はPythonコードのみを含むMarkdownコードブロック形式にしてください。説明、挨拶、その他のテキストは一切含めないでください。"
        "ファイルが存在しない場合は新規作成し、存在する場合は内容を適切に上書きまたは修正してください。"
        "出力はファイルに書き込むコードそのものでなければなりません。"
    )
    
    readme_content = read_file(os.path.join(PROJECT_DIR, "README.md"))
    
    # タスクからファイル名を抽出。見つからなければフォールバックを使用
    match = re.search(r'`([^`]+)`', task)
    if match:
        target_file = match.group(1)
    elif fallback_filename:
        print(f"⚠️ タスクにファイル名がありませんでした。フォールバックファイル `{fallback_filename}` を使用します。")
        target_file = fallback_filename
    else:
        print(f"❌ タスク「{task}」から対象ファイル名が見つけられず、フォールバックもありません。スキップします。")
        return False
        
    target_filepath = os.path.join(PROJECT_DIR, target_file)
    existing_code = read_file(target_filepath)
    
    user_prompt = (
        f"あなたは Engineer #{engineer_id} です。以下のタスクを厳密に実行してください。\n\n"
        f"## プロジェクト全体のREADME:\n---\n{readme_content}\n---\n\n"
        f"## あなたが担当するタスク:\n- {task}\n\n"
        f"## 対象ファイル: `{target_file}`\n\n"
        f"## 現在のファイルの内容:\n```python\n{existing_code}\n```\n\n"
        f"上記の情報を元に、タスクを完了させるための`{target_file}`の完全なコードを、Markdownコードブロック形式で生成してください。"
    )
    
    print(f"\n===== 👷 Engineer AI #{engineer_id}'s Turn on: {task} =====")
    code = ai_call(system_prompt, user_prompt)

    if code:
        code = re.sub(r'^```[a-zA-Z]*\n', '', code)
        code = re.sub(r'\n```$', '', code)
        write_file(target_filepath, code)
        print(f"✅ Engineer #{engineer_id}が `{target_filepath}` を更新しました。")
        return True
    else:
        print(f"❌ Engineer #{engineer_id}がコードの生成に失敗しました。")
        return False

# --- メインワークフロー ---

def main():
    """メインの実行関数。"""
    print("ようこそ！組織的AIコーディングシステム (Hugging Face版)へ。")
    
    # プロジェクトのクリーンアップ確認
    if os.path.exists(PROJECT_DIR) and os.listdir(PROJECT_DIR):
        print(f"\n⚠️ 警告: '{PROJECT_DIR}' ディレクトリには既にファイルが存在します。")
        while True:
            choice = input("開始前に中身を全て削除しますか？ (y/n): ").lower().strip()
            if choice in ['y', 'yes']:
                print(f"🧹 '{PROJECT_DIR}' ディレクトリの中身を削除しています...")
                clean_project_dir()
                print("✅ 削除が完了しました。")
                break
            elif choice in ['n', 'no']:
                print("📂 既存のファイルを保持して処理を続行します。")
                break
            else:
                print("無効な入力です。'y' または 'n' を入力してください。")
    
    if not client:
        print("エラー: クライアントが初期化されていません。")
        return

    user_request = read_file(REQUEST_FILE)
    if not user_request:
        print(f"エラー: 開発要求ファイル '{REQUEST_FILE}' が見つからないか、内容が空です。")
        return
    print(f"\n📄 '{REQUEST_FILE}' から開発要求を読み込みました:\n---\n{user_request}\n---")
    
    create_project_dir()
    
    president_instruction = president_ai(user_request)
    if not president_instruction:
        print("Presidentが指示を出せませんでした。処理を中断します。")
        return
        
    if not project_manager_ai(president_instruction):
        print("Project Managerがタスク計画を立てられませんでした。処理を中断します。")
        return

    engineer_id_counter = 1
    main_filename = None

    # タスクがなくなるまでループ
    while True:
        readme_content = read_file(os.path.join(PROJECT_DIR, "README.md"))
        tasks = re.findall(r'-\s*\[\s*\]\s*(.*)', readme_content)

        if not tasks:
            print("\n🎉 全てのタスクが完了しました！")
            break

        current_task_text = tasks[0]
        
        # メインファイル名をまだ特定していない場合、READMEから推定する
        if not main_filename:
            all_tasks_in_readme = re.findall(r'-\s*\[[\s|x]\]\s*(.*)', readme_content)
            for t in all_tasks_in_readme:
                match = re.search(r'`([^`]+)`', t)
                if match:
                    main_filename = match.group(1)
                    print(f"💡 プロジェクトのメインファイルを `{main_filename}` と推定しました。")
                    break
        
        # Engineerを交代で割り当て、タスクを実行
        engineer_id = (engineer_id_counter - 1) % 2 + 1
        success = engineer_ai(current_task_text, engineer_id, fallback_filename=main_filename)
        engineer_id_counter += 1

        # READMEのタスクを更新
        if success:
            current_task_line = f"- [ ] {current_task_text}"
            new_readme_content = readme_content.replace(current_task_line, f"- [x] {current_task_text}", 1)
            write_file(os.path.join(PROJECT_DIR, "README.md"), new_readme_content)
            print(f"✅ タスクを完了済みに更新: {current_task_text}")
        else:
            print(f"❌ タスクの処理に失敗しました。処理を中断します: {current_task_text}")
            break
        
        time.sleep(1)

    print("\n===== 最終的なプロジェクト構成 =====")
    for root, _, files in os.walk(PROJECT_DIR):
        for name in files:
            print(os.path.join(root, name).replace('\\', '/'))
    print("====================================")
    print("開発を終了します。")

if __name__ == "__main__":
    main()