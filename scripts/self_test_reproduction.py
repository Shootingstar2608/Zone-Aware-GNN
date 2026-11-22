"""
Self-Test Reproduction Pipeline (with Automated Environment Lifecycle)
======================================================================
Kịch bản tự động hóa kiểm định độc lập quy trình tái lập (Reproduction Guide) từ đầu đến cuối:
1. Tự động kiểm tra / khởi tạo môi trường Conda độc lập `dp4` (Python 3.10).
2. Tự động cài đặt toàn bộ thư viện từ `requirements.txt` vào `dp4`.
3. Tạo thư mục tạm `self_test/` và Clone sạch mã nguồn từ GitHub chính thức:
   https://github.com/Shootingstar2608/Zone-Aware-GNN.git (nhánh feat/member-d-data-cards).
4. Kiểm tra dữ liệu tĩnh (hcm_osrm_dataset.csv, zone_labels.csv).
5. Build HCM-Sim v2 (28 ngày, t-in 12, t-out 24).
6. Kiểm định Assertion tập dữ liệu (S=2,653, N=17, Tin=12, Tout=24).
7. Đóng băng giao thức thực nghiệm (freeze protocol --force).
8. Sinh phân vùng Non-IID & Xác thực bit-for-bit (verify_partitions.py).
9. Kiểm định nguyên lý nhân quả (Pytest 52 unit tests).
10. Thực thi Smoke Test kiểm thử nhanh (clean_smoke_v1.csv).
11. Dọn dẹp: Tự động xóa sạch thư mục `self_test/` và tự động xóa môi trường `dp4`.
"""

import os
import sys
import shutil
import subprocess
import time
from pathlib import Path

# Đảm bảo Windows console và child processes luôn dùng UTF-8 chuẩn
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

REPO_URL = "https://github.com/Shootingstar2608/Zone-Aware-GNN.git"
BRANCH = "feat/member-d-data-cards"
SELF_TEST_DIR = Path("self_test").resolve()
CLONE_DIR = SELF_TEST_DIR / "Zone-Aware-GNN"
REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_FILE = REPO_ROOT / "requirements.txt"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def force_remove_directory(target_path: Path):
    if not target_path.exists():
        return
    import stat
    def on_rm_error(func, path_str, exc_info):
        try:
            os.chmod(path_str, stat.S_IWRITE)
            func(path_str)
        except Exception:
            pass
    for _ in range(3):
        try:
            shutil.rmtree(str(target_path), onerror=on_rm_error)
            if not target_path.exists():
                return
        except Exception:
            time.sleep(0.5)
    if target_path.exists():
        subprocess.run(f'cmd.exe /c "rmdir /s /q \"{target_path}\""', shell=True, capture_output=True)


def find_conda_cmd():
    conda_exe = shutil.which("conda")
    if conda_exe:
        return conda_exe
    common_paths = [
        Path(r"C:\Users\Admin\miniconda3\condabin\conda.bat"),
        Path(r"C:\Users\Admin\miniconda3\Scripts\conda.exe"),
        Path(os.path.expanduser(r"~\miniconda3\condabin\conda.bat")),
        Path(os.path.expanduser(r"~\miniconda3\Scripts\conda.exe")),
    ]
    for p in common_paths:
        if p.exists():
            return str(p)
    return "conda"


def find_env_python(env_name="dp4"):
    common_pythons = [
        Path(rf"C:\Users\Admin\miniconda3\envs\{env_name}\python.exe"),
        Path(os.path.expanduser(rf"~\miniconda3\envs\{env_name}\python.exe")),
        Path(os.path.expanduser(rf"~\anaconda3\envs\{env_name}\python.exe")),
    ]
    for p in common_pythons:
        if p.exists():
            return p
    return None


def run_step(cmd, cwd, step_name):
    log(f"--- BẮT ĐẦU: {step_name} ---")
    log(f"CMD: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    res = subprocess.run(
        cmd,
        cwd=str(cwd),
        shell=isinstance(cmd, str),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    elapsed = time.time() - t0
    if res.returncode != 0:
        log(f"❌ THẤT BẠI: {step_name} (sau {elapsed:.2f}s)")
        print("STDOUT:\n", res.stdout)
        print("STDERR:\n", res.stderr)
        raise RuntimeError(f"Step failed with code {res.returncode}: {step_name}")
    log(f"✅ THÀNH CÔNG: {step_name} ({elapsed:.2f}s)")
    return res.stdout


def ensure_environment(env_name="dp4"):
    conda_cmd = find_conda_cmd()
    env_dir = Path(r"C:\Users\Admin\miniconda3\envs") / env_name
    env_python = find_env_python(env_name)

    if env_python is None or not env_python.exists():
        if env_dir.exists():
            log(f"Phát hiện thư mục `{env_dir}` dở dang. Đang xóa sạch cưỡng chế...")
            force_remove_directory(env_dir)

        log(f"Môi trường `{env_name}` chưa tồn tại. Đang tạo mới với Python 3.10...")
        run_step(
            f'"{conda_cmd}" create -n {env_name} python=3.10 -y',
            cwd=REPO_ROOT,
            step_name=f"Khởi tạo môi trường Conda {env_name} (Python 3.10)"
        )
        env_python = find_env_python(env_name)
        if env_python is None or not env_python.exists():
            raise RuntimeError(f"Không tìm thấy Python executable của `{env_name}` sau khi tạo!")

    # Cài đặt / Đồng bộ thư viện từ requirements.txt
    log(f"Kiểm tra và cài đặt thư viện từ requirements.txt vào môi trường `{env_name}`...")
    run_step(
        [str(env_python), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)],
        cwd=REPO_ROOT,
        step_name=f"Cài đặt requirements.txt vào {env_name}"
    )

    # Xác thực PyTorch và Pandas hoạt động đúng chuẩn
    run_step(
        [
            str(env_python), "-c",
            "import torch, pandas, scipy, pytest; "
            "print(f'✅ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()}), Pandas {pandas.__version__}')"
        ],
        cwd=REPO_ROOT,
        step_name=f"Xác thực môi trường {env_name}"
    )

    return env_python


def main(cleanup_dir=True, cleanup_env=True):
    print("=" * 70)
    print("🚀 BẮT ĐẦU TỰ ĐỘNG HÓA KIỂM ĐỊNH TÁI LẬP (SELF-TEST REPRODUCTION)")
    print(f"URL: {REPO_URL} (Branch: {BRANCH})")
    print(f"Thư mục kiểm thử: {SELF_TEST_DIR}")
    print("=" * 70)

    conda_cmd = find_conda_cmd()
    target_python = None

    try:
        # Bước 0: Quản lý vòng đời môi trường Conda độc lập dp4
        target_python = ensure_environment(env_name="dp4")
        log(f"Môi trường thực thi chính thức: {target_python}")

        # 1. Chuẩn bị thư mục self_test
        log("Chuẩn bị thư mục self_test...")
        force_remove_directory(SELF_TEST_DIR)
        SELF_TEST_DIR.mkdir(parents=True, exist_ok=True)

        # 2. Clone từ GitHub
        run_step(
            ["git", "clone", "--branch", BRANCH, "--depth", "1", REPO_URL, str(CLONE_DIR)],
            cwd=SELF_TEST_DIR,
            step_name=f"Git Clone từ {REPO_URL} (Branch {BRANCH})"
        )

        # 3. Kiểm tra dữ liệu tĩnh
        osrm_csv = CLONE_DIR / "data" / "raw" / "hcm_osrm_dataset.csv"
        zone_csv = CLONE_DIR / "data" / "raw" / "zone_labels.csv"
        assert osrm_csv.exists(), f"Thiếu {osrm_csv}"
        assert zone_csv.exists(), f"Thiếu {zone_csv}"
        log("✅ Kiểm tra dữ liệu tĩnh (hcm_osrm_dataset.csv, zone_labels.csv): PASS")

        # 4. Build HCM-Sim v2 (28 ngày, 15 phút)
        run_step(
            [
                str(target_python), "scripts/dev/generate_synthetic_traffic.py",
                "--seed", "42",
                "--days", "28",
                "--start", "2026-10-05",
                "--interval-min", "15",
                "--out", "data/generated/hcm_sim_v2/hcm_sim_traffic.csv"
            ],
            cwd=CLONE_DIR,
            step_name="Bước 1.1: Sinh dữ liệu tổng hợp HCM-Sim v2 (28 ngày)"
        )

        # 5. Dựng đồ thị tensor chuỗi thời gian
        run_step(
            [
                str(target_python), "scripts/build_graph.py",
                "--traffic-path", "data/generated/hcm_sim_v2/hcm_sim_traffic.csv",
                "--out-dir", "data/processed/hcm_sim_v2",
                "--t-in", "12",
                "--t-out", "24"
            ],
            cwd=CLONE_DIR,
            step_name="Bước 1.2: Dựng đồ thị tensor chuỗi thời gian (build_graph --t-in 12 --t-out 24)"
        )

        # 6. Kiểm định Assertions
        check_code = (
            "import json, torch; "
            "from pathlib import Path; "
            "p = Path('data/processed/hcm_sim_v2'); "
            "meta = json.loads((p / 'meta.json').read_text(encoding='utf-8')); "
            "ds = torch.load(p / 'graph_dataset.pt', map_location='cpu', weights_only=False); "
            "assert meta['data_source'] == 'synthetic', meta; "
            "assert meta['N'] == 17, meta; "
            "assert meta['T_in'] == 12 and meta['T_out'] == 24, meta; "
            "assert meta['S'] == 2653, meta; "
            "assert ds['X'].shape[0] == 2653 and ds['Y'].shape[-1] == 24; "
            "print('Dataset assertion: S=2653, N=17, Tin=12, Tout=24 -> 100% OK!')"
        )
        out = run_step(
            [str(target_python), "-c", check_code],
            cwd=CLONE_DIR,
            step_name="Bước 1.3: Kiểm định Assertions (S=2,653, N=17, T_in=12, T_out=24)"
        )
        print(out.strip())

        # 7. Đóng băng giao thức thực nghiệm (Freeze Protocol)
        run_step(
            [str(target_python), "scripts/freeze_hcm_sim_protocol.py", "--data-dir", "data/processed/hcm_sim_v2", "--force"],
            cwd=CLONE_DIR,
            step_name="Bước 2: Đóng băng giao thức thực nghiệm (freeze_hcm_sim_protocol.py)"
        )
        protocol_json = CLONE_DIR / "data" / "processed" / "hcm_sim_v2" / "benchmark_protocol_v1.json"
        assert protocol_json.exists(), f"Thiếu {protocol_json}"

        # 8. Sinh phân vùng Non-IID
        run_step(
            [str(target_python), "-m", "benchmark.partition_gen", "--data-dir", "data/processed/hcm_sim_v2"],
            cwd=CLONE_DIR,
            step_name="Bước 3.1: Sinh phân vùng Non-IID (partition_gen.py)"
        )

        # 9. Xác thực tái lập bit-for-bit
        out = run_step(
            [str(target_python), "scripts/verify_partitions.py", "--data-dir", "data/processed/hcm_sim_v2"],
            cwd=CLONE_DIR,
            step_name="Bước 3.2: Xác thực tái lập bit-for-bit (verify_partitions.py)"
        )
        print(out.strip())

        # 10. Kiểm định nguyên lý nhân quả (Pytest)
        run_step(
            [
                str(target_python), "-m", "pytest", "-q",
                "tests/test_temporal_benchmark.py",
                "tests/test_benchmark.py",
                "tests/test_eval_protocol.py",
                "tests/test_hcm_real_static_metadata.py"
            ],
            cwd=CLONE_DIR,
            step_name="Bước 4: Kiểm định nguyên lý nhân quả (52 unit tests)"
        )

        # 11. Thực thi Smoke Run nhanh
        (CLONE_DIR / "data" / "results" / "hcm_sim_v2").mkdir(parents=True, exist_ok=True)
        run_step(
            [
                str(target_python), "scripts/run_multi_seed.py",
                "--data-dir", "data/processed/hcm_sim_v2",
                "--protocol", "data/processed/hcm_sim_v2/benchmark_protocol_v1.json",
                "--models", "persistence", "mean_by_node", "seasonal_naive",
                "--split-modes", "chrono",
                "--seeds", "42",
                "--epochs", "1",
                "--out", "data/results/hcm_sim_v2/clean_smoke_v1.csv"
            ],
            cwd=CLONE_DIR,
            step_name="Bước 5: Thực thi Smoke Run kiểm thử nhanh"
        )

        smoke_csv = CLONE_DIR / "data" / "results" / "hcm_sim_v2" / "clean_smoke_v1.csv"
        assert smoke_csv.exists(), f"Thiếu {smoke_csv}"
        log("✅ Tệp kết quả Smoke Run tồn tại và hợp lệ!")

        print("\n" + "=" * 70)
        print("🎉 TOÀN BỘ QUY TRÌNH TÁI LẬP CHẠY THÀNH CÔNG 100% TỪ CLONE SẠCH TRÊN DP4!")
        print("=" * 70 + "\n")

    finally:
        # Dọn dẹp thư mục self_test
        if cleanup_dir and SELF_TEST_DIR.exists():
            log("Dọn dẹp xóa thư mục self_test theo yêu cầu...")
            force_remove_directory(SELF_TEST_DIR)
            log("✅ Đã dọn dẹp sạch sẽ self_test.")

        # Dọn dẹp xóa môi trường conda dp4
        if cleanup_env:
            log("Dọn dẹp xóa môi trường Conda `dp4` theo yêu cầu...")
            subprocess.run(f'"{conda_cmd}" env remove -n dp4 -y', shell=True)
            env_dir = Path(r"C:\Users\Admin\miniconda3\envs\dp4")
            if env_dir.exists():
                force_remove_directory(env_dir)
            log("✅ Đã xóa hoàn toàn môi trường Conda `dp4` sạch sẽ!")


if __name__ == "__main__":
    # Tham số dòng lệnh:
    # --keep-dir: không xóa thư mục self_test
    # --keep-env: không xóa môi trường dp4
    # Mặc định: tự động xóa cả thư mục self_test và môi trường dp4 sau khi chạy
    do_cleanup_dir = "--keep-dir" not in sys.argv and "--keep" not in sys.argv
    do_cleanup_env = "--keep-env" not in sys.argv and "--keep" not in sys.argv
    main(cleanup_dir=do_cleanup_dir, cleanup_env=do_cleanup_env)
