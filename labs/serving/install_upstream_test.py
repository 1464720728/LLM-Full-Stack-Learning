"""Copy one course-owned test to an exact, independent vLLM source checkout.

Dry-run unless --apply. Refuses overwrite and a mismatched source revision.
It does not install dependencies, execute tests or change upstream production code.
"""
import argparse
from pathlib import Path
import subprocess
from .source_walk import SNAPSHOTS


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--apply',action='store_true');a=p.parse_args();root=a.root.resolve()
    sha=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
    if sha!=SNAPSHOTS['vllm']['commit']:p.error('checkout differs from the verified source snapshot')
    target=root/'tests/v1/core/test_course_scheduler_contract.py'
    if not target.parent.is_dir():p.error('not the expected vLLM source layout')
    if target.exists():p.error('target already exists; inspect it rather than overwriting')
    source=Path(__file__).parent/'upstream_tests/test_course_scheduler_contract.py'
    print(f'Source: {source}\nTarget: {target}\nCommit: {sha}')
    if not a.apply:print('Dry run; add --apply to copy the new test.');return
    target.write_bytes(source.read_bytes());print('Copied one new test file; no upstream implementation changed.')
if __name__=='__main__':main()
