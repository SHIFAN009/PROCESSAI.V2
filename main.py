import os
import asyncio
import logging
import uuid
import time
import secrets
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any

from fastapi import FastAPI, HTTPException, Depends, Security, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from pydantic import BaseModel, Field, field_validator

from sqlalchemy import create_engine, Column, String, DateTime, Float, Integer, Text, Boolean, Index, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session

import redis.asyncio as aioredis
import httpx
from openai import AsyncOpenAI
from semantic_scholar import SemanticScholar
import numpy as np
from sentence_transformers import SentenceTransformer
from passlib.context import CryptContext
import jwt
import bleach
from pint import UnitRegistry
from pythonjsonlogger import jsonlogger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chemeai")
handler = logging.StreamHandler()
handler.setFormatter(jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
logger.addHandler(handler)

class Config:
    SECRET_KEY: str = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/chemeai")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    BING_SEARCH_KEY: str = os.getenv("BING_SEARCH_KEY", "")
    SEMANTIC_SCHOLAR_KEY: str = os.getenv("SEMANTIC_SCHOLAR_KEY", "")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    ALGORITHM: str = os.getenv("JWT_ALG", "HS256")
    ACCESS_TOKEN_EXPIRE_MIN: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MIN", "60"))
    LIMIT_FREE: str = os.getenv("LIMIT_FREE", "30/minute")
    LIMIT_PRO: str = os.getenv("LIMIT_PRO", "120/minute")
    LIMIT_ENT: str = os.getenv("LIMIT_ENT", "300/minute")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2000"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))
    OPENAI_TIMEOUT_S: float = float(os.getenv("OPENAI_TIMEOUT_S", "30"))
    OPENAI_RETRIES: int = int(os.getenv("OPENAI_RETRIES", "2"))
    ALLOW_ORIGINS: List[str] = os.getenv("ALLOW_ORIGINS", "https://yourdomain.com").split(",")
    TRUSTED_HOSTS: List[str] = os.getenv("TRUSTED_HOSTS", "yourdomain.com,*.yourdomain.com,localhost,127.0.0.1").split(",")
    MAX_QUERY_LENGTH: int = int(os.getenv("MAX_QUERY_LENGTH", "2000"))

config = Config()

# Database setup
engine = create_engine(config.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis setup - initialize later to avoid connection issues during startup
redis = None

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

try:
    ureg = UnitRegistry()
except Exception as e:
    logger.warning(f"Failed to initialize UnitRegistry: {e}")
    ureg = None

class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=lambda: secrets.token_urlsafe(16))
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    subscription_tier = Column(String, default="free")
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    api_key = Column(String, unique=True, index=True, nullable=True)

class Query(Base):
    __tablename__ = "queries"
    id = Column(String, primary_key=True, default=lambda: secrets.token_urlsafe(16))
    user_id = Column(String, index=True)
    query_text = Column(Text)
    response_text = Column(Text)
    accuracy_score = Column(Float)
    response_time = Column(Float)
    sources_used = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String)

# Create indexes
Index("ix_queries_user_created", Query.user_id, Query.created_at)

# Create tables
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    logger.error(f"Failed to create database tables: {e}")

class ChatRequest(BaseModel):
    message: str = Field(...)
    conversation_history: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    
    @field_validator("message")
    @classmethod
    def validate_message(cls, v):
        if not isinstance(v, str):
            v = str(v)
        v = v.strip()
        if not v:
            raise ValueError("Message cannot be empty")
        if len(v) > config.MAX_QUERY_LENGTH:
            raise ValueError(f"Message too long. Max {config.MAX_QUERY_LENGTH} characters")
        return bleach.clean(v)

class SourceOut(BaseModel):
    title: str
    url: str
    type: str

class ChatResponse(BaseModel):
    response: str
    accuracy_score: float
    sources: List[SourceOut]
    response_time: float
    query_id: str

class LoginRequest(BaseModel):
    email: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class SignupRequest(BaseModel):
    email: str
    password: str
    tier: Optional[str] = "free"

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user(
    request: Request, 
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security), 
    db: Session = Depends(get_db)
) -> User:
    # Check for API key first
    api_key = request.headers.get("X-API-Key")
    if api_key:
        user = db.query(User).filter(User.api_key == api_key, User.is_active == True).first()
        if not user:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return user
    
    # Check for JWT token
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing credentials")
    
    token = credentials.credentials
    try:
        payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        user_id: str = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
        if not user:
            raise HTTPException(status_code=401, detail="User not found or inactive")
        return user
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def parse_limit(limit: str) -> int:
    try:
        num, unit = limit.split("/")
        return int(num)
    except Exception:
        return 30

def tier_limit_count(tier: str) -> int:
    if tier == "enterprise":
        return parse_limit(config.LIMIT_ENT)
    if tier == "professional":
        return parse_limit(config.LIMIT_PRO)
    return parse_limit(config.LIMIT_FREE)

async def enforce_rate_limit(user: User, request: Request):
    global redis
    if redis is None:
        return  # Skip rate limiting if Redis is not available
        
    now = int(time.time())
    window = now // 60
    limit = tier_limit_count(user.subscription_tier or "free")
    
    # Use user IP if available, otherwise use user ID only
    client_host = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
    key = f"rl:{user.id}:{window}:{client_host}"
    
    try:
        cur = await redis.incr(key)
        if cur == 1:
            await redis.expire(key, 65)
        if cur > limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
    except Exception as e:
        logger.warning(f"Rate limit check failed: {e}")
        # Continue without rate limiting if Redis fails

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = rid
        start = time.time()
        
        try:
            response = await call_next(request)
            duration = time.time() - start
            logger.info({
                "event": "request",
                "rid": rid,
                "path": request.url.path,
                "method": request.method,
                "status": response.status_code,
                "duration_ms": int(duration * 1000)
            })
            response.headers["X-Request-ID"] = rid
            return response
        except Exception as e:
            logger.exception({"event": "request_error", "rid": rid, "error": str(e)})
            raise

class ChemEKnowledgeBase:
    def __init__(self):
        self.chemical_properties = {
            "antoine_coefficients": {
                "water": {"A": 8.07131, "B": 1730.63, "C": 233.426},
                "benzene": {"A": 6.90565, "B": 1211.033, "C": 220.790},
                "toluene": {"A": 6.95464, "B": 1344.800, "C": 219.482},
            },
            "critical_properties": {
                "water": {"Tc": 647.1, "Pc": 220.64, "omega": 0.345},
                "benzene": {"Tc": 562.05, "Pc": 48.95, "omega": 0.210},
            },
        }
        self.equipment_data = {
            "heat_exchanger_U_values": {
                "water_water": 500,
                "steam_water": 1000,
                "oil_water": 300,
            },
            "pipe_roughness": {
                "commercial_steel": 4.5e-5,
                "pvc": 1.5e-6,
                "stainless_steel": 1.5e-5,
            },
        }
        self.safety_data = {
            "flash_points": {"benzene": -11, "toluene": 4, "methanol": 11},
            "exposure_limits": {
                "benzene": {"TWA": 1, "STEL": 5}, 
                "toluene": {"TWA": 50, "STEL": 150}
            },
        }

@dataclass
class SearchResult:
    title: str
    content: str
    url: str
    relevance_score: float
    source_type: str

class AdvancedSearchEngine:
    def __init__(self):
        self.bing_key = config.BING_SEARCH_KEY
        try:
            self.semantic_scholar = SemanticScholar(
                api_key=config.SEMANTIC_SCHOLAR_KEY
            ) if config.SEMANTIC_SCHOLAR_KEY else SemanticScholar()
        except Exception as e:
            logger.warning(f"Failed to initialize Semantic Scholar: {e}")
            self.semantic_scholar = None

    @staticmethod
    def _cache_key(prefix: str, query: str) -> str:
        import hashlib
        h = hashlib.sha256(query.encode("utf-8")).hexdigest()
        return f"{prefix}:{h}"

    @staticmethod
    def _is_reliable_source(url: str) -> bool:
        domains = [
            "nist.gov", "aiche.org", "asme.org", "api.org", 
            "engineeringtoolbox.com", "che.com", "chemengonline.com",
            "wikipedia.org", "sciencedirect.com", "springer.com"
        ]
        u = (url or "").lower()
        return any(d in u for d in domains)

    async def search_web(self, query: str) -> List[SearchResult]:
        global redis
        if redis is None:
            return []
            
        key = self._cache_key("web", query)
        try:
            cached = await redis.get(key)
            if cached:
                return [SearchResult(**x) for x in json.loads(cached)]
        except Exception:
            pass  # Continue without cache
            
        results: List[SearchResult] = []
        
        if not self.bing_key:
            return results
            
        try:
            headers = {"Ocp-Apim-Subscription-Key": self.bing_key}
            params = {
                "q": f"{query} chemical engineering", 
                "count": 5, 
                "safeSearch": "Strict"
            }
            
            async with httpx.AsyncClient(timeout=10.0) as session:
                resp = await session.get(
                    "https://api.cognitive.microsoft.com/bing/v7.0/search", 
                    headers=headers, 
                    params=params
                )
                
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("webPages", {}).get("value", []):
                    url = item.get("url", "")
                    if self._is_reliable_source(url):
                        results.append(SearchResult(
                            title=item.get("name", ""),
                            content=item.get("snippet", ""),
                            url=url,
                            relevance_score=0.8,
                            source_type="web"
                        ))
        except Exception as e:
            logger.error({"event": "web_search_error", "error": str(e)})
            
        # Try to cache results
        try:
            if redis:
                await redis.set(key, json.dumps([r.__dict__ for r in results]), ex=3600 * 10)
        except Exception:
            pass
            
        return results

    async def search_academic(self, query: str) -> List[SearchResult]:
        global redis
        if redis is None or self.semantic_scholar is None:
            return []
            
        key = self._cache_key("acad", query)
        try:
            cached = await redis.get(key)
            if cached:
                return [SearchResult(**x) for x in json.loads(cached)]
        except Exception:
            pass
            
        results: List[SearchResult] = []
        
        try:
            papers = self.semantic_scholar.search_paper(query, limit=5)
            for p in papers:
                if hasattr(p, 'abstract') and p.abstract and len(p.abstract) > 50:
                    results.append(SearchResult(
                        title=p.title or "Untitled",
                        content=p.abstract[:600],
                        url=getattr(p, 'url', '') or "",
                        relevance_score=0.9,
                        source_type="academic"
                    ))
        except Exception as e:
            logger.error({"event": "academic_search_error", "error": str(e)})
            
        # Try to cache results
        try:
            if redis:
                await redis.set(key, json.dumps([r.__dict__ for r in results]), ex=3600 * 24)
        except Exception:
            pass
            
        return results

class AccuracyValidator:
    def __init__(self):
        self.kb = ChemEKnowledgeBase()
        try:
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning(f"Failed to load sentence transformer: {e}")
            self.embedder = None
            
    def _has_consistent_units(self, text: str) -> bool:
        if ureg is None:
            return False
            
        unit_tokens = re.findall(r"\b([A-Za-z]+(?:/[A-Za-z]+|\^[0-9]+)?)\b", text)
        for token in unit_tokens[:50]:
            try:
                _ = 1 * ureg(token)
                return True
            except Exception:
                continue
        return False
        
    def _has_reasonable_values(self, text: str) -> bool:
        nums = re.findall(r"\b\d+\.?\d*\b", text)
        for ns in nums[:200]:
            try:
                v = float(ns)
                if v > 1e10 or (0 < v < 1e-10):
                    return False
            except Exception:
                continue
        return True
        
    def _mentions_safety(self, text: str) -> bool:
        safety_keywords = ["safety", "hazard", "flammable", "explosive", "tox", "pressure", "ppe"]
        t = text.lower()
        return any(k in t for k in safety_keywords)
        
    def _semantic_relevance(self, query: str, response: str) -> float:
        if self.embedder is None:
            return 0.5  # Default score if embedder is not available
            
        try:
            q = self.embedder.encode([query])[0]
            r = self.embedder.encode([response])[0]
            sim = float(np.dot(q, r) / (np.linalg.norm(q) * np.linalg.norm(r) + 1e-12))
            return max(0.0, min(1.0, (sim + 1) / 2))
        except Exception:
            return 0.5
            
    def score(self, query: str, response: str) -> float:
        base = 0.6
        
        if self._has_consistent_units(response):
            base += 0.15
        if self._has_reasonable_values(response):
            base += 0.1
        if self._mentions_safety(response):
            base += 0.05
            
        base += 0.1 * self._semantic_relevance(query, response)
        
        return float(min(1.0, base))

class ChemEAI:
    def __init__(self):
        self.search = AdvancedSearchEngine()
        self.validator = AccuracyValidator()
        self.client = AsyncOpenAI(
            api_key=config.OPENAI_API_KEY, 
            timeout=config.OPENAI_TIMEOUT_S
        ) if config.OPENAI_API_KEY else None
        self.system_prompt = self._system_prompt()
        
    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are ChemE AI, a professional chemical engineering assistant with 90%+ accuracy.\n"
            "CRITICAL REQUIREMENTS:\n"
            "1. Provide step-by-step calculations with proper SI units.\n"
            "2. Include safety considerations (PPE, pressure/temperature limits).\n"
            "3. Reference standards where relevant (ASME/API/AIChE).\n"
            "4. Validate dimensional consistency using units.\n"
            "5. Provide realistic numbers and a sanity check.\n"
            "6. State assumptions clearly.\n"
            "7. Include equipment selection rationale.\n\n"
            "CALCULATION FORMAT:\n"
            "1) Given data with units\n"
            "2) Assumptions\n"
            "3) Step-by-step solution\n"
            "4) Final answer with units\n"
            "5) Sanity check\n"
        )
        
    async def process_query(self, query: str, user: User, db: Session) -> ChatResponse:
        t0 = time.time()
        
        # Search for relevant information
        try:
            web_task = asyncio.create_task(self.search.search_web(query))
            acad_task = asyncio.create_task(self.search.search_academic(query))
            web_results, acad_results = await asyncio.gather(web_task, acad_task)
        except Exception as e:
            logger.error(f"Search error: {e}")
            web_results, acad_results = [], []
            
        results = (web_results or []) + (acad_results or [])
        context = self._build_context(query, results)
        
        # Generate response
        content = await self._generate(query, context)
        
        # Calculate accuracy and metrics
        score = self.validator.score(query, content)
        dt = time.time() - t0
        
        # Log the query
        qid = self._log_query(db, user.id, query, content, score, dt, results)
        
        return ChatResponse(
            response=content,
            accuracy_score=score,
            sources=[
                SourceOut(title=r.title, url=r.url, type=r.source_type) 
                for r in results[:5]
            ],
            response_time=dt,
            query_id=qid
        )
        
    def _build_context(self, query: str, results: List[SearchResult]) -> str:
        ctx = [f"User Query: {query}", "\nRelevant Information:"]
        
        for i, r in enumerate(results[:6], 1):
            ctx.append(
                f"\n{i}. {r.title}\n"
                f"Source: {r.source_type}\n"
                f"Content: {r.content[:400]}..."
            )
            
        return "\n".join(ctx)
        
    async def _generate(self, query: str, context: str) -> str:
        if self.client is None:
            return "AI service is not configured. Please check API key configuration."
            
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"{context}\n\nPlease provide a rigorous answer to: {query}"},
        ]
        
        for attempt in range(config.OPENAI_RETRIES + 1):
            try:
                resp = await self.client.chat.completions.create(
                    model=config.MODEL_NAME,
                    messages=messages,
                    max_tokens=config.MAX_TOKENS,
                    temperature=config.TEMPERATURE
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                if attempt >= config.OPENAI_RETRIES:
                    logger.error({"event": "openai_error", "error": str(e)})
                    return "I encountered an issue generating the response. Please retry."
                await asyncio.sleep(0.5 * (attempt + 1))
                
        return "Unexpected error occurred."
        
    def _log_query(self, db: Session, user_id: str, q: str, r: str, score: float, dt: float, sources: List[SearchResult]) -> str:
        try:
            rec = Query(
                user_id=user_id,
                query_text=q,
                response_text=r,
                accuracy_score=score,
                response_time=dt,
                sources_used=json.dumps([
                    {"title": s.title, "url": s.url, "type": s.source_type} 
                    for s in sources
                ]),
                ip_address=None
            )
            db.add(rec)
            db.commit()
            return rec.id
        except Exception as e:
            logger.error(f"Failed to log query: {e}")
            return "unknown"

# Initialize the AI system
cheme_ai = ChemEAI()

# FastAPI app setup
app = FastAPI(
    title="ChemE AI API",
    description="Production-grade Chemical Engineering AI Assistant",
    version="2.0.1",
    docs_url="/docs" if config.ENVIRONMENT != "production" else None,
    redoc_url=None
)

# Add middleware
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=config.TRUSTED_HOSTS)

@app.on_event("startup")
async def startup_event():
    """Initialize Redis connection on startup"""
    global redis
    try:
        redis = aioredis.from_url(config.REDIS_URL, decode_responses=True)
        await redis.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.warning(f"Failed to connect to Redis: {e}")
        redis = None

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up Redis connection on shutdown"""
    global redis
    if redis:
        try:
            await redis.close()
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")

@app.post("/api/auth/signup", response_model=LoginResponse)
async def signup(body: SignupRequest, db: Session = Depends(get_db)):
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == body.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    user = User(
        email=body.email,
        hashed_password=pwd_context.hash(body.password),
        subscription_tier=body.tier or "free",
        api_key=secrets.token_urlsafe(24)
    )
    db.add(user)
    db.commit()
    
    # Generate JWT token
    token_data = {
        "sub": user.id,
        "exp": datetime.utcnow() + timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MIN)
    }
    token = jwt.encode(token_data, config.SECRET_KEY, algorithm=config.ALGORITHM)
    
    return LoginResponse(access_token=token)

@app.post("/api/auth/login", response_model=LoginResponse)
async def login(body: LoginRequest, db: Session = Depends(get_db)):
    # Find user
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not pwd_context.verify(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if not user.is_active:
        raise HTTPException(status_code=403, detail="User inactive")
    
    # Update last active
    user.last_active = datetime.utcnow()
    db.commit()
    
    # Generate JWT token
    token_data = {
        "sub": user.id,
        "exp": datetime.utcnow() + timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MIN)
    }
    token = jwt.encode(token_data, config.SECRET_KEY, algorithm=config.ALGORITHM)
    
    return LoginResponse(access_token=token)

@app.get("/api/auth/apikey")
async def get_api_key(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user.api_key:
        current_user.api_key = secrets.token_urlsafe(24)
        db.commit()
    return {"api_key": current_user.api_key}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: Request,
    chat_request: ChatRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Enforce rate limiting
    await enforce_rate_limit(current_user, request)
    
    content = chat_request.message.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        resp = await cheme_ai.process_query(content, current_user, db)
        background_tasks.add_task(update_user_activity, current_user.id)
        return resp
    except Exception as e:
        logger.exception({
            "event": "chat_error",
            "rid": getattr(request.state, 'request_id', None),
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail="Processing error")

@app.get("/api/health")
async def health_check(db: Session = Depends(get_db)):
    status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.1"
    }
    
    # Check Redis
    try:
        if redis:
            pong = await redis.ping()
            status["redis"] = "ok" if pong else "down"
        else:
            status["redis"] = "not_connected"
    except Exception as e:
        status["redis"] = f"error: {e}"
    
    # Check database
    try:
        db.execute(text("SELECT 1"))
        status["db"] = "ok"
    except Exception as e:
        status["db"] = f"error: {e}"
    
    # Check OpenAI configuration
    status["model"] = "configured" if bool(config.OPENAI_API_KEY) else "missing_api_key"
    
    return status

async def update_user_activity(user_id: str):
    """Background task to update user's last active timestamp"""
    db = None
    try:
        db = SessionLocal()
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            user.last_active = datetime.utcnow()
            db.commit()
    except Exception as e:
        logger.error({"event": "update_activity_error", "error": str(e)})
    finally:
        if db:
            try:
                db.close()
            except Exception:
                pass

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    rid = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "rid": rid,
            "ts": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", None)
    logger.exception({
        "event": "unhandled_exception",
        "rid": rid,
        "error": str(exc)
    })
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "rid": rid,
            "ts": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=(config.ENVIRONMENT == "development"),
        log_level="info"
    )
