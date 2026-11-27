"""Compare parent-head reuse, damped retry and tail-aware selection with a fixed expert."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import subprocess
import sys

import benchmark
import contest_submission as harness

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'work/v5/controller'
ROUTES = {'base':(False,1.,0.),'carry':(True,1.,0.),'retry':(False,.25,0.),
          'carry_retry':(True,.25,0.),'tail':(True,1.,.35),'combined':(True,.25,.35)}
ROUTES.update(origin=(False,1.,0.), origin_carry=(True,1.,0.), origin_retry=(True,.25,0.))
CASES = [('trace',n) for n in (0,3,5,8,10)] + [('drift',n) for n in (0,5,8)] + [('fresh',n) for n in (3,7)] + [('official',n) for n in (1,6)]

def single(route, setting, split, expert):
    if split not in range(11):
        raise ValueError('Controller studies use development schedules only')
    if expert=='v4':
        spec=importlib.util.spec_from_file_location('engine',HERE.parent/'V4/engine.py')
        mod=importlib.util.module_from_spec(spec)
        sys.modules['engine']=mod
        spec.loader.exec_module(mod)
    harness.ENGINE_CARRY,harness.ENGINE_RETRY_SCALE,harness.ENGINE_TAIL_WEIGHT=ROUTES[route]
    harness.ENGINE_ORIGIN_AWARE=route.startswith("origin")
    benchmark.OUT=OUT/expert/route
    benchmark.run('C','v5',split,setting,profile=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('route',choices=['screen',*ROUTES])
    parser.add_argument('--routes',nargs='+',choices=ROUTES,default=list(ROUTES))
    parser.add_argument('--expert',choices=['v4','v5'],default='v4')
    parser.add_argument('--setting',choices=['trace','drift','fresh','official'])
    parser.add_argument('--split',type=int)
    parser.add_argument('--workers',type=int,default=9)
    a=parser.parse_args()
    if a.route!='screen':
        return single(a.route,a.setting,a.split,a.expert)
    def run(spec):
        route,setting,split=spec
        dst=OUT/a.expert/route
        dst.mkdir(exist_ok=True,parents=True)
        with (dst/f'{setting}-{split}.log').open('w') as log:
            subprocess.run([sys.executable,'-B',__file__,route,'--expert',a.expert,'--setting',setting,'--split',str(split)],stdout=log,stderr=log,check=True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(run,[(r,s,n) for r in a.routes for s,n in CASES]))

if __name__=='__main__':
    main()
