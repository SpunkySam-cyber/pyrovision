"""Shared strict YAML loading that rejects silently overwritten keys."""

from __future__ import annotations

from typing import Any

import yaml

from .errors import ConfigurationError


class UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConfigurationError(f"Duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_unique_yaml(text: str) -> Any:
    try:
        return yaml.load(text, Loader=UniqueKeyLoader)
    except ConfigurationError:
        raise
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML: {exc}") from exc
