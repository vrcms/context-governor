"""store_root from a config FILE is anchored to that file's directory.

The 2026-08-03 incident. `store_root = "./contextstore"` in
integration/governor.toml was resolved with os.path.abspath() against whatever
directory the launcher started in, so ONE config file addressed TWO stores:

    run-governor.ps1 invoked from integration/   ->  integration/contextstore  (90 notes)
    governor-guard.vbs (workDir = repo root)     ->  ./contextstore            (empty)

The operator reset one store and the governor silently began writing to the
other — on two consecutive days. The damage is not the wasted directory: it is
that every "corpus 0" baseline taken that way was unverifiable, and a recall
corpus believed to be reset could still be serving slices mined from another
project.

A state directory whose identity depends on the caller's working directory is
the same machine-specific coupling this codebase removes everywhere else.
"""

from __future__ import annotations

import os

from contextmanager.launcher import resolve_config

_FLAG_FIELDS = [
    "upstream_base_url", "listen_host", "listen_port", "store_root",
    "handle_threshold_tokens", "handle_threshold_ratio", "context_budget_ratio",
    "stub_preview_chars", "rehydrate_budget_tokens", "request_timeout",
    "model_alias", "diff_min_similarity", "diff_lookback", "diff_max_chars",
    "tokenize_max_chars",
]


def _opts(**over) -> dict:
    base = {k: None for k in (["config", "provider", "cli"] + _FLAG_FIELDS)}
    base["dry_run"] = False
    base.update(over)
    return base


def _file_cfg(store_root: str) -> dict:
    return {"proxy": {"store_root": store_root}}


class TestTheRegression:
    def test_one_config_file_addresses_one_store_from_any_cwd(self, tmp_path, monkeypatch):
        """THE BUG. Same config file, two different working directories — the
        resolved store must be identical. Before the fix these differed."""
        cfg_path = str(tmp_path / "integration" / "governor.toml")
        os.makedirs(os.path.dirname(cfg_path), exist_ok=True)

        elsewhere = tmp_path / "some" / "other" / "cwd"
        elsewhere.mkdir(parents=True)

        monkeypatch.chdir(tmp_path)
        from_root = resolve_config(_opts(), _file_cfg("./contextstore"), cfg_path)

        monkeypatch.chdir(elsewhere)
        from_elsewhere = resolve_config(_opts(), _file_cfg("./contextstore"), cfg_path)

        assert from_root.store_root == from_elsewhere.store_root
        assert os.path.isabs(from_root.store_root)

    def test_it_anchors_to_the_config_file_not_the_cwd(self, tmp_path, monkeypatch):
        """And it anchors to the RIGHT place: next to the config file."""
        cfg_dir = tmp_path / "integration"
        cfg_dir.mkdir()
        cfg_path = str(cfg_dir / "governor.toml")

        monkeypatch.chdir(tmp_path)  # repo root, NOT the config's directory
        cfg = resolve_config(_opts(), _file_cfg("./contextstore"), cfg_path)

        assert cfg.store_root == os.path.normpath(str(cfg_dir / "contextstore"))
        assert cfg.store_root != os.path.normpath(str(tmp_path / "contextstore"))


class TestWhatMustNotChange:
    def test_an_absolute_store_root_is_left_alone(self, tmp_path):
        absolute = os.path.abspath(str(tmp_path / "explicit-store"))
        cfg = resolve_config(_opts(), _file_cfg(absolute),
                             str(tmp_path / "governor.toml"))
        assert cfg.store_root == absolute

    def test_an_explicit_cli_flag_keeps_shell_semantics(self, tmp_path, monkeypatch):
        """A path typed in a terminal means "relative to where I am standing".
        Layer 3 applies after the anchoring, so the flag still wins."""
        cfg_dir = tmp_path / "integration"
        cfg_dir.mkdir()
        monkeypatch.chdir(tmp_path)

        cfg = resolve_config(_opts(store_root="./cli-store"),
                             _file_cfg("./contextstore"),
                             str(cfg_dir / "governor.toml"))
        assert cfg.store_root == "./cli-store"

    def test_no_config_path_leaves_the_value_untouched(self, tmp_path):
        """Back-compat: callers that do not pass config_path (including every
        existing test) must see exactly the old behaviour."""
        cfg = resolve_config(_opts(), _file_cfg("./contextstore"))
        assert cfg.store_root == "./contextstore"

    def test_a_config_without_store_root_is_unaffected(self, tmp_path):
        cfg = resolve_config(_opts(), {"proxy": {"listen_port": 8901}},
                             str(tmp_path / "governor.toml"))
        assert cfg.listen_port == 8901
        assert cfg.store_root == "./contextstore"  # the ProxyConfig default


class TestVisibility:
    def test_the_resolved_store_is_reported_in_metrics(self, tmp_path):
        """The silent switch is the real damage. /metrics must name the store
        absolutely, so a reset the governor never saw is distinguishable from a
        reset that worked."""
        from starlette.testclient import TestClient
        from contextmanager.proxy.app import create_app
        from contextmanager.proxy.config import ProxyConfig

        store_dir = str(tmp_path / "store")
        cfg = ProxyConfig(
            upstream_base_url="http://upstream.test",
            store_root=store_dir,
            listen_host="127.0.0.1",
            listen_port=8900,
        )
        with TestClient(create_app(cfg)) as client:
            body = client.get("/metrics").json()
        assert body["store_root"] == os.path.abspath(store_dir)
