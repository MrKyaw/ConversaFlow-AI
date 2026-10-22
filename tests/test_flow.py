"""Tests for flow schema validation."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from conversaflow.dialogue.flow import Flow


def test_flow_model_validates_real_flows():
    from conversaflow.data_loader import load_flows

    flows = load_flows(Path("data/flows"))
    assert len(flows) == 8


def test_duplicate_node_ids():

    flow_json = json.loads(Path("data/flows/order_status.json").read_text())
    flow_json["nodes"][1]["id"] = flow_json["nodes"][0]["id"]
    with pytest.raises(ValidationError):
        Flow.model_validate(flow_json)


def test_unknown_start():
    flow_json = json.loads(Path("data/flows/order_status.json").read_text())
    flow_json["start"] = "nope"
    with pytest.raises(ValidationError):
        Flow.model_validate(flow_json)


def test_unreachable_node():
    payload = json.loads(Path("data/flows/order_status.json").read_text())
    payload["nodes"].append({"id": "unreachable", "type": "end"})
    with pytest.raises(ValidationError, match="unreachable nodes"):
        Flow.model_validate(payload)
