"""Local transformer learning lab. Start: python -m uvicorn main:app --host 127.0.0.1 --port 8000

Run one worker. Model loading is lazy; educational toy endpoints require no
checkpoint download. All instrumentation requests share a re-entrant lock.
"""
import threading
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from math_core import toy_attention, toy_training

ROOT=Path(__file__).resolve().parent
LOCK=threading.RLock()
_engine=None
app=FastAPI(title="Transformer Learning Lab",version="2.1.0")
app.add_middleware(GZipMiddleware,minimum_size=1000)
app.add_middleware(TrustedHostMiddleware,allowed_hosts=["127.0.0.1","localhost","[::1]","testserver"])


def get_engine():
    global _engine
    if _engine is None:
        from engine import Engine
        _engine=Engine()
    return _engine


def run(method,*args,**kwargs):
    with LOCK:
        try:
            return getattr(get_engine(),method)(*args,**kwargs)
        except ValueError as exc:
            raise HTTPException(422,str(exc)) from exc
        except (ImportError,OSError) as exc:
            raise HTTPException(503,"Model unavailable. Check installed requirements and local checkpoint/cache. "+str(exc)) from exc


class Prompt(BaseModel):
    prompt:str=Field(min_length=1,max_length=20000)


class Inspect(Prompt):
    layer:int=Field(ge=1,le=100)
    head:int=Field(ge=1,le=100)
    position:int|None=Field(default=None,ge=0,le=255)
    target_id:int|None=Field(default=None,ge=0)


class Generate(Prompt):
    n_tokens:int=Field(default=12,ge=1,le=64)
    temperature:float=Field(default=1,ge=0,le=5,allow_inf_nan=False)
    runs:int=Field(default=1,ge=1,le=5)
    seed:int|None=Field(default=None,ge=0,le=2**63-2)
    use_cache:bool=False
    stop_eos:bool=True


class Intervention(Prompt):
    layer:int=Field(ge=1,le=100)
    scope:Literal["last","all"]="last"
    target_id:int|None=Field(default=None,ge=0)
    contrast_id:int|None=Field(default=None,ge=0)
    uniform:bool=False


class Compare(BaseModel):
    prompt_a:str=Field(min_length=1,max_length=20000)
    prompt_b:str=Field(min_length=1,max_length=20000)
    target_ids:list[int]=Field(default_factory=list,max_length=20)
    layer:int|None=Field(default=None,ge=1,le=100)
    head:int=Field(default=1,ge=1,le=100)


class ProbeFit(BaseModel):
    train_prompts:list[str]=Field(min_length=4,max_length=64)
    test_prompts:list[str]=Field(min_length=2,max_length=32)
    ridge:float=Field(default=1,gt=0,le=100,allow_inf_nan=False)


class Jacobian(Prompt):
    layer:int=Field(ge=1,le=100)
    coordinate:int=Field(default=0,ge=0)
    epsilon:float=Field(default=0.01,ge=0.0001,le=1,allow_inf_nan=False)


@app.get("/api/health")
def health(load:bool=False):
    with LOCK:
        if not load and _engine is None:
            return {"status":"ready","model_loaded":False,"app_version":"2.1.0"}
        try:
            return {"status":"ok","model_loaded":True,"provenance":get_engine().provenance}
        except (ValueError,ImportError,OSError) as exc:
            raise HTTPException(503,str(exc)) from exc


@app.get("/api/tokenize")
def tokenize(prompt:str=Query("",max_length=20000)):
    with LOCK:
        try:
            e=get_engine()
            ids=e.tokenizer.encode(prompt,add_special_tokens=False)
            return {"prompt":prompt,"tokens":e.tokens(ids),"add_special_tokens":False}
        except (ValueError,ImportError,OSError) as exc:
            raise HTTPException(503,str(exc)) from exc


@app.get("/api/decode")
def decode(id:int=Query(...,ge=0)):
    with LOCK:
        e=get_engine()
        if id>=e.vocab:
            raise HTTPException(422,"Vocabulary ID out of range")
        return {"id":id,"text":e.tokenizer.decode([id])}


@app.post("/api/predict")
def predict(request:Prompt): return run("predict",**request.model_dump())


@app.post("/api/generate")
def generate(request:Generate): return run("generate",**request.model_dump())


@app.post("/api/attention")
def attention(request:Prompt): return run("attention",**request.model_dump())


@app.post("/api/inspect")
def inspect(request:Inspect): return run("inspect",**request.model_dump())


@app.post("/api/lenses")
def lenses(request:Prompt): return run("lenses",**request.model_dump())


@app.post("/api/intervene")
def intervene(request:Intervention): return run("knockout",**request.model_dump())


@app.post("/api/compare")
def compare(request:Compare): return run("compare",**request.model_dump())


@app.post("/api/training-trace")
def training_trace(request:Prompt): return run("training_trace",**request.model_dump())


@app.post("/api/probe-fit")
def probe_fit(request:ProbeFit):
    if any(not s or len(s)>20000 for s in request.train_prompts+request.test_prompts):
        raise HTTPException(422,"Probe prompts must contain 1 to 20,000 characters")
    return run("fit_probe",**request.model_dump())


@app.post("/api/jacobian")
def jacobian(request:Jacobian): return run("jacobian",**request.model_dump())


@app.post("/api/verify")
def verify(request:Prompt): return run("verify",**request.model_dump())


@app.get("/api/toy-attention")
def toy(): return toy_attention()


@app.get("/api/toy-training")
def train_toy(steps:int=Query(1,ge=1,le=50),learning_rate:float=Query(0.2,gt=0,le=1)):
    return toy_training(steps,learning_rate)


app.mount("/",StaticFiles(directory=ROOT/"static",html=True),name="static")
