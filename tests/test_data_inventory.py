"""
Tests for Semantic Memory — data inventory loading.

Covers:
  - load_data_inventory() writes exactly 4 systems into the store.
  - Each system is stored under the correct namespace ('inventory', 'systems').
  - Each system's store key matches its 'id' field.
  - Required fields (id, name, data_categories, retention_period, description)
    are present in every stored value.
  - The four expected system ids are present.
  - load_data_inventory() is idempotent: calling it twice does not create duplicates.
  - The module-level singleton store is pre-populated at import time.
  - load_data_inventory() accepts an injected store and writes into it instead.
"""

from __future__ import annotations

import pytest
from langgraph.store.memory import InMemoryStore

from dsar_langgraph_agent.triage_graph import (
    _INVENTORY_NAMESPACE,
    load_data_inventory,
    store as singleton_store,
)

_EXPECTED_IDS = {"crm", "marketing_hub", "zendesk", "production_sql"}
_REQUIRED_FIELDS = {"id", "name", "data_categories", "retention_period", "description"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_store_with_inventory() -> InMemoryStore:
    s = InMemoryStore()
    load_data_inventory(target_store=s)
    return s


# ---------------------------------------------------------------------------
# 1. Basic loading
# ---------------------------------------------------------------------------

class TestInventoryLoading:

    def test_four_systems_loaded(self):
        s = _fresh_store_with_inventory()
        items = s.search(_INVENTORY_NAMESPACE)
        assert len(items) == 4

    def test_all_expected_ids_present(self):
        s = _fresh_store_with_inventory()
        items = s.search(_INVENTORY_NAMESPACE)
        stored_keys = {i.key for i in items}
        assert _EXPECTED_IDS == stored_keys

    def test_correct_namespace(self):
        s = _fresh_store_with_inventory()
        items = s.search(_INVENTORY_NAMESPACE)
        for item in items:
            assert item.namespace == _INVENTORY_NAMESPACE

    def test_required_fields_in_every_system(self):
        s = _fresh_store_with_inventory()
        items = s.search(_INVENTORY_NAMESPACE)
        for item in items:
            missing = _REQUIRED_FIELDS - set(item.value.keys())
            assert not missing, f"System {item.key!r} missing fields: {missing}"

    def test_key_matches_id_field(self):
        s = _fresh_store_with_inventory()
        items = s.search(_INVENTORY_NAMESPACE)
        for item in items:
            assert item.key == item.value["id"]

    def test_data_categories_is_a_list(self):
        s = _fresh_store_with_inventory()
        items = s.search(_INVENTORY_NAMESPACE)
        for item in items:
            assert isinstance(item.value["data_categories"], list)
            assert len(item.value["data_categories"]) > 0

    def test_retention_period_is_nonempty_string(self):
        s = _fresh_store_with_inventory()
        items = s.search(_INVENTORY_NAMESPACE)
        for item in items:
            assert isinstance(item.value["retention_period"], str)
            assert item.value["retention_period"].strip()

    def test_description_is_nonempty_string(self):
        s = _fresh_store_with_inventory()
        items = s.search(_INVENTORY_NAMESPACE)
        for item in items:
            assert isinstance(item.value["description"], str)
            assert item.value["description"].strip()


# ---------------------------------------------------------------------------
# 2. Spot-checks on individual systems
# ---------------------------------------------------------------------------

class TestSystemContent:

    def setup_method(self):
        self.s = _fresh_store_with_inventory()

    def _get(self, system_id: str) -> dict:
        item = self.s.get(_INVENTORY_NAMESPACE, system_id)
        assert item is not None, f"System {system_id!r} not found in store"
        return item.value

    def test_crm_name(self):
        assert self._get("crm")["name"] == "CRM"

    def test_crm_has_email_and_address(self):
        cats = self._get("crm")["data_categories"]
        assert "email" in cats
        assert "postal_address" in cats

    def test_marketing_hub_name(self):
        assert self._get("marketing_hub")["name"] == "Marketing_Hub"

    def test_marketing_hub_has_consent_field(self):
        cats = self._get("marketing_hub")["data_categories"]
        assert any("consent" in c for c in cats)

    def test_zendesk_name(self):
        assert self._get("zendesk")["name"] == "Zendesk"

    def test_zendesk_has_ticket_content(self):
        cats = self._get("zendesk")["data_categories"]
        assert any("ticket" in c for c in cats)

    def test_production_sql_name(self):
        assert self._get("production_sql")["name"] == "Production_SQL"

    def test_production_sql_has_purchase_history(self):
        cats = self._get("production_sql")["data_categories"]
        assert "purchase_history" in cats


# ---------------------------------------------------------------------------
# 3. Idempotency
# ---------------------------------------------------------------------------

class TestIdempotency:

    def test_double_load_does_not_duplicate(self):
        s = InMemoryStore()
        load_data_inventory(target_store=s)
        load_data_inventory(target_store=s)
        items = s.search(_INVENTORY_NAMESPACE)
        assert len(items) == 4  # not 8

    def test_return_value_is_list_of_four_ids(self):
        s = InMemoryStore()
        result = load_data_inventory(target_store=s)
        assert set(result) == _EXPECTED_IDS
        assert len(result) == 4


# ---------------------------------------------------------------------------
# 4. Module-level pre-load (singleton store)
# ---------------------------------------------------------------------------

class TestModuleLevelPreload:

    def test_singleton_store_has_inventory_at_import(self):
        """The module-level store should already contain the inventory."""
        items = singleton_store.search(_INVENTORY_NAMESPACE)
        stored_keys = {i.key for i in items}
        # All four systems must be present (there may also be episode entries)
        assert _EXPECTED_IDS.issubset(stored_keys)

    def test_singleton_store_systems_have_required_fields(self):
        items = singleton_store.search(_INVENTORY_NAMESPACE)
        for item in items:
            if item.key in _EXPECTED_IDS:
                missing = _REQUIRED_FIELDS - set(item.value.keys())
                assert not missing, f"Singleton store: {item.key!r} missing {missing}"
