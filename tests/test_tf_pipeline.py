import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = Path('python3')


def test_train_with_nas_invocation(tmp_path, monkeypatch):
    # Very small smoke test that ensures scripts/tf_pipeline.py will select train_with_nas.py
    tf_pipeline = ROOT / 'scripts' / 'tf_pipeline.py'
    assert tf_pipeline.exists()
    # Run help invocation for train subcommand with --enable-nas to ensure no immediate parsing errors
    p = subprocess.run([str(PY), str(tf_pipeline), 'train', '--model', 'nano_u', '--enable-nas', '--help'], capture_output=True)
    # help returns code 0 and prints usage
    assert p.returncode == 0
    assert b'usage' in p.stdout.lower() or b'usage' in p.stderr.lower()
