import os
import sys
import subprocess
import importlib
from pathlib import Path

REQUIRED_PACKAGES = {
    "PyMuPDF": {"import_name": "fitz", "package": "PyMuPDF"},
    "Pillow": {"import_name": "PIL", "package": "Pillow"},
    "llama_cloud": {"import_name": "llama_cloud", "package": "llama-cloud"},
    "Pydantic": {"import_name": "pydantic", "package": "pydantic"},
    "Streamlit": {"import_name": "streamlit", "package": "streamlit"},
    "dotenv": {"import_name": "dotenv", "package": "python-dotenv"},
}

ENV_PATH = Path(__file__).resolve().parent / ".env"


def install_package(package_name: str) -> bool:
    print(f"Đang cài package: {package_name}...")
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "install", package_name],
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        print(f"Cài đặt thành công: {package_name}")
        return True
    print(f"Lỗi cài {package_name}: {completed.stderr.strip()}")
    return False


def check_package(name: str, info: dict) -> tuple[str, bool, str]:
    import_name = info["import_name"]
    package_name = info["package"]
    try:
        importlib.import_module(import_name)
        return name, True, "Đã cài"
    except ImportError:
        installed = install_package(package_name)
        status = "Đã cài" if installed else "Thiếu"
        note = "Cài thành công" if installed else "Không cài được package"
        return name, installed, note


def check_python_version() -> tuple[str, bool, str]:
    required = (3, 10)
    current = sys.version_info
    if current >= required:
        return "Python", True, f"{current.major}.{current.minor}.{current.micro}"
    return "Python", False, f"Phiên bản cần >= 3.10, hiện tại {current.major}.{current.minor}.{current.micro}"


def ensure_env_file() -> tuple[str, bool, str]:
    if ENV_PATH.exists():
        with ENV_PATH.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        keys = [line.split("=", 1)[0].strip() for line in lines if "=" in line]
        if "LLAMA_CLOUD_API_KEY" in keys:
            return ".env", True, "Tìm thấy khóa LLAMA_CLOUD_API_KEY (không hiển thị giá trị)."
        return ".env", False, "File .env tồn tại nhưng chưa khai báo LLAMA_CLOUD_API_KEY"

    ENV_PATH.write_text("LLAMA_CLOUD_API_KEY='KEY CỦA BẠN'\n", encoding="utf-8")
    return ".env", False, "Đã tạo file .env mẫu; hãy cập nhật giá trị LLAMA_CLOUD_API_KEY của bạn."


def print_table(rows: list[tuple[str, bool, str]]) -> None:
    header = f"{'Kiểm tra':<20} {'PASS/FAIL':<10} Ghi chú"
    print("\n" + header)
    print("-" * len(header))
    for name, ok, note in rows:
        status = "PASS" if ok else "FAIL"
        print(f"{name:<20} {status:<10} {note}")


def load_dotenv_key() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=ENV_PATH)
        if os.getenv("LLAMA_CLOUD_API_KEY"):
            print("LLAMA_CLOUD_API_KEY đã được nạp từ .env. (Không hiển thị giá trị)")
        else:
            print("LLAMA_CLOUD_API_KEY chưa được thiết lập hoặc có giá trị trống.")
    except Exception:
        print("Không thể nạp .env; hãy kiểm tra python-dotenv.")


def main() -> None:
    print("Kiểm tra môi trường OCR cho Buổi 5")
    results = [check_python_version()]
    for name, info in REQUIRED_PACKAGES.items():
        results.append(check_package(name, info))

    env_result = ensure_env_file()
    results.append(env_result)

    print_table(results)
    load_dotenv_key()
    print("\nLưu ý: Không in giá trị trực tiếp của LLAMA_CLOUD_API_KEY để bảo mật.")


if __name__ == "__main__":
    main()
