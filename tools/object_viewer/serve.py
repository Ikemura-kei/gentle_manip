"""Run from any directory: python3 tools/object_viewer/serve.py [--port 8765]."""
import argparse
from dataclasses import asdict
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gentle_manip.assets.registry import OBJECT_MAP

FOOD = {'tofu', 'mushroom', 'cherry_tomato', 'tomato', 'banana_chunk',
        'pasta_bundle', 'raspberry', 'banana', 'strawberry'}

# Names registered by the object-expansion pipeline (source_and_filter -> mesh_prep ->
# register_object) are recorded here so the atlas can group them separately from the
# hand-curated food/dev sets; see gentle_manip/scripts/object_expansion/EXPANSION_LOG.csv.
EXPANSION_LOG = ROOT / 'gentle_manip' / 'scripts' / 'object_expansion' / 'EXPANSION_LOG.csv'

def expansion_names():
    if not EXPANSION_LOG.exists():
        return {}
    import csv
    out = {}
    with open(EXPANSION_LOG) as f:
        for row in csv.DictReader(f):
            out[row['name']] = row
    return out

def catalog():
    items = []
    exp = expansion_names()
    for name, obj in OBJECT_MAP.items():
        category = re.sub(r'\d+$', '', name)
        if name in exp:
            group = 'Object expansion'
        elif category in FOOD:
            group = 'Food'
        else:
            group = 'Development / synthetic'
        path = Path(obj.mesh_path) if obj.mesh_path else None
        items.append(dict(name=name, category=category, group=group,
            mesh='/' + path.relative_to(ROOT).as_posix() if path and path.exists() else None,
            missing=bool(path and not path.exists()), size=obj.size,
            material=asdict(obj.material), object_type=obj.object_type,
            parked=category == 'raspberry',
            expansion_meta=exp.get(name)))
    return items

def find_runs(name: str, limit: int = 6):
    """Most recent collect_demos_synth_v4 run dirs for single_lift_<name>_soft, with stats
    and served-relative paths to videos / final-grasp PNGs."""
    import yaml
    base = ROOT / 'dataset' / 'demos' / f'single_lift_{name}_soft'
    if not base.exists():
        return []
    run_dirs = sorted([d for d in base.iterdir() if d.is_dir()], key=lambda d: d.name, reverse=True)
    out = []
    for run_dir in run_dirs[:limit]:
        stats_path = run_dir / 'stats.yaml'
        stats = yaml.safe_load(stats_path.read_text()) if stats_path.exists() else None
        vids, pngs = [], []
        vdir = run_dir / 'videos'
        if vdir.exists():
            for f in sorted(vdir.iterdir()):
                rel = '/' + f.relative_to(ROOT).as_posix()
                (vids if f.suffix == '.mp4' else pngs if f.suffix == '.png' else []).append(rel) \
                    if f.suffix in ('.mp4', '.png') else None
        out.append(dict(run_dir=run_dir.name, stats=stats, videos=vids[:8], images=pngs[:8]))
    return out

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/objects':
            data = json.dumps(catalog()).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif parsed.path.startswith('/api/runs/'):
            name = unquote(parsed.path[len('/api/runs/'):])
            data = json.dumps(find_runs(name)).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == '/':
            self.send_response(302)
            self.send_header('Location', '/tools/object_viewer/index.html')
            self.end_headers()
        else:
            super().do_GET()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    print(f'Object viewer: http://localhost:{args.port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, directory=str(ROOT))).serve_forever()
