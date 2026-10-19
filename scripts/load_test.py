"""Send sample traffic to the API so Prometheus/Grafana have something to show.

Usage:  python scripts/load_test.py --url http://localhost:8000 --n 200
"""
import argparse
import random
import time

import requests

SAMPLE = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2,
          "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--delay", type=float, default=0.1)
    args = parser.parse_args()
    ok = 0
    for i in range(args.n):
        payload = dict(SAMPLE)
        payload.update(age=random.randint(30, 75), chol=random.randint(150, 350),
                       thalach=random.randint(100, 200), oldpeak=round(random.uniform(0, 4), 1),
                       cp=random.randint(1, 4), exang=random.randint(0, 1),
                       ca=random.randint(0, 3), thal=random.choice([3, 6, 7]))
        if i % 25 == 0:  # occasionally send an invalid request to show error metrics
            payload["sex"] = 9
        resp = requests.post(f"{args.url}/predict", json=payload, timeout=10)
        ok += resp.status_code == 200
        time.sleep(args.delay)
    print(f"sent {args.n} requests, {ok} successful")


if __name__ == "__main__":
    main()
