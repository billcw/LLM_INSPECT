"""Run the real checkpoint acceptance checks and save an honest runtime report.

Run AFTER installing requirements and intentionally downloading/loading the model.
Uses the same LLM_LAB_MODEL / REVISION / OFFLINE settings as the app.
"""
import argparse
import json
from pathlib import Path
from engine import Engine


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt',default='The capital of Japan is called')
    parser.add_argument('--output',default='model_verification.json')
    args=parser.parse_args()
    report=Engine().verify(args.prompt)
    # Refuse accidental overwrite of an existing experiment report.
    with Path(args.output).open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {report['passed_count']}/{report['check_count']}; {args.output}")
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':main()
