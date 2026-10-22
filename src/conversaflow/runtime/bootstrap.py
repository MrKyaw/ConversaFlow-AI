"""Wire the composition root.

One function builds every component exactly once. Tests either call this with a
temporary settings object or construct a smaller runtime by hand; the API layer
only ever sees the resulting object graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..agents.planner import Planner
from ..agents.registry import AgentRouter
from ..agents.tools import ToolRegistry, build_tool_registry
from ..config import Settings
from ..data_loader import load_flows, load_knowledge
from ..dialogue.flow import Flow
from ..dialogue.policy import DialoguePolicy
from ..dialogue.runner import FlowRunner
from ..knowledge.bm25 import KnowledgeBase
from ..nlu.classifier import IntentModel, TrainingReport
from ..nlu.pipeline import NluConfig, NluPipeline
from ..nlu.train import load_or_train
from ..observability.metrics import MetricsRegistry
from ..voice.ssml import compose
from .engine import ConversationEngine
from .stores import InMemorySessionStore, SessionStore, SqliteSessionStore


@dataclass
class Runtime:
    settings: Settings
    model: IntentModel
    model_report: TrainingReport | None
    nlu: NluPipeline
    flows: dict[str, Flow]
    knowledge: KnowledgeBase
    tools: ToolRegistry
    router: AgentRouter
    store: SessionStore
    metrics: MetricsRegistry
    policy: DialoguePolicy
    runner: FlowRunner
    planner: Planner
    engine: ConversationEngine
    tenants: list[str] = field(default_factory=list)

    def synthesize(self, text: str, locale: str = "en-US") -> str:
        return compose(text, locale=locale)


def build_runtime(
    settings: Settings | None = None,
    *,
    store: SessionStore | None = None,
    model: IntentModel | None = None,
) -> Runtime:
    resolved = settings or Settings.from_env()
    data_dir = resolved.data_dir
    flows = load_flows(data_dir / "flows")

    model_report: TrainingReport | None = None
    if model is None:
        model, model_report = load_or_train(
            resolved.model_path, data_dir / "intents" / "train.jsonl"
        )

    knowledge = KnowledgeBase()
    documents = load_knowledge(data_dir / "kb" / "faq.jsonl")
    tenants = sorted({principal.tenant_id for principal in resolved.api_keys.values()} | {"demo"})
    if documents:
        for tenant in tenants:
            knowledge.ingest(tenant, documents)

    tools = build_tool_registry(timeout=resolved.tool_timeout_seconds)
    nlu = NluPipeline(
        model,
        NluConfig(
            intent_threshold=resolved.intent_threshold,
            margin_threshold=resolved.margin_threshold,
        ),
    )
    router = AgentRouter()
    policy = DialoguePolicy(
        flows,
        knowledge,
        kb_min_score=resolved.kb_min_score,
        intent_threshold=resolved.intent_threshold,
    )
    runner = FlowRunner(flows, tools)
    planner = Planner(flows, router, tools)
    metrics = MetricsRegistry()
    if store is None:
        store = _build_store(resolved)
    engine = ConversationEngine(
        settings=resolved,
        nlu=nlu,
        flows=flows,
        runner=runner,
        policy=policy,
        knowledge=knowledge,
        tools=tools,
        router=router,
        store=store,
        metrics=metrics,
    )
    return Runtime(
        settings=resolved,
        model=model,
        model_report=model_report,
        nlu=nlu,
        flows=flows,
        knowledge=knowledge,
        tools=tools,
        router=router,
        store=store,
        metrics=metrics,
        policy=policy,
        runner=runner,
        planner=planner,
        engine=engine,
        tenants=tenants,
    )


def _build_store(settings: Settings) -> SessionStore:
    if settings.store == "memory":
        return InMemorySessionStore()
    return SqliteSessionStore(Path(settings.db_path))
