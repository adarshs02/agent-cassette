# Adapted from Project Blackbox (https://github.com/alejandro-publius/blackbox-datahub),
# Apache-2.0. Modified for Retrace.
"""Push the pipeline's schemas, docs, owners, tags, and lineage into DataHub."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from retrace.config import Settings
from retrace.datahub.catalog import (
    DATASET_DOCS,
    FIELD_DOCS,
    INCIDENT_TAG,
    OWNERS,
    PLATFORM,
    TABLES,
    TAGS,
    dataset_urn,
    lineage_edges,
)
from retrace.pipeline.workspace import Workspace


def _models() -> Any:
    try:
        import datahub.metadata.schema_classes as models
    except ImportError as error:
        raise ImportError("install the ingest extra: uv sync --extra datahub") from error
    return models


def _field_type(models: Any, native: str) -> Any:
    upper = native.upper()
    if any(t in upper for t in ("DOUBLE", "DECIMAL", "INT", "FLOAT")):
        return models.SchemaFieldDataTypeClass(type=models.NumberTypeClass())
    if upper == "DATE":
        return models.SchemaFieldDataTypeClass(type=models.DateTypeClass())
    if "TIMESTAMP" in upper:
        return models.SchemaFieldDataTypeClass(type=models.TimeTypeClass())
    return models.SchemaFieldDataTypeClass(type=models.StringTypeClass())


def build_proposals(warehouse: Path) -> list[Any]:
    models = _models()
    from datahub.emitter.mcp import MetadataChangeProposalWrapper as MCPW

    proposals: list[Any] = []
    for user, (display, title, _tables) in OWNERS.items():
        proposals.append(
            MCPW(
                entityUrn=f"urn:li:corpuser:{user}",
                aspect=models.CorpUserInfoClass(
                    active=True, displayName=display, title=title, email=f"{user}@retrace.demo"
                ),
            )
        )
    owner_of = {t: u for u, (_, _, tables) in OWNERS.items() for t in tables}
    upstreams: dict[str, list[str]] = {}
    for up, down in lineage_edges():
        upstreams.setdefault(down, []).append(up)
    # Tag entities must exist before GlobalTags/add_tags can reference them; emit
    # them before any dataset proposal. Preserve first-seen order for determinism.
    all_tags = list(dict.fromkeys([INCIDENT_TAG, *(tag for tags in TAGS.values() for tag in tags)]))
    for tag in all_tags:
        proposals.append(
            MCPW(
                entityUrn=f"urn:li:tag:{tag}",
                aspect=models.TagPropertiesClass(
                    name=tag, description=f"Retrace-managed tag: {tag}."
                ),
            )
        )
    con = duckdb.connect(str(warehouse), read_only=True)
    try:
        for table in TABLES:
            urn = dataset_urn(table)
            columns = con.execute(f"DESCRIBE {table}").fetchall()
            docs = FIELD_DOCS.get(table, {})
            proposals.append(
                MCPW(
                    entityUrn=urn,
                    aspect=models.DatasetPropertiesClass(
                        name=table,
                        description=DATASET_DOCS.get(table, ""),
                        customProperties={"pipeline": "retrace-demo-retail"},
                    ),
                )
            )
            proposals.append(
                MCPW(
                    entityUrn=urn,
                    aspect=models.SchemaMetadataClass(
                        schemaName=table,
                        platform=f"urn:li:dataPlatform:{PLATFORM}",
                        version=0,
                        hash="",
                        platformSchema=models.OtherSchemaClass(rawSchema=""),
                        fields=[
                            models.SchemaFieldClass(
                                fieldPath=name,
                                type=_field_type(models, native),
                                nativeDataType=native,
                                description=docs.get(name, ""),
                            )
                            for name, native, *_ in columns
                        ],
                    ),
                )
            )
            if table in owner_of:
                proposals.append(
                    MCPW(
                        entityUrn=urn,
                        aspect=models.OwnershipClass(
                            owners=[
                                models.OwnerClass(
                                    owner=f"urn:li:corpuser:{owner_of[table]}",
                                    type=models.OwnershipTypeClass.TECHNICAL_OWNER,
                                )
                            ]
                        ),
                    )
                )
            # Always emit GlobalTags (empty when untagged) so tags a previous run
            # added, such as retrace-incident, are cleared.
            proposals.append(
                MCPW(
                    entityUrn=urn,
                    aspect=models.GlobalTagsClass(
                        tags=[
                            models.TagAssociationClass(tag=f"urn:li:tag:{tag}")
                            for tag in TAGS.get(table, [])
                        ]
                    ),
                )
            )
            if table in upstreams:
                proposals.append(
                    MCPW(
                        entityUrn=urn,
                        aspect=models.UpstreamLineageClass(
                            upstreams=[
                                models.UpstreamClass(
                                    dataset=dataset_urn(u),
                                    type=models.DatasetLineageTypeClass.TRANSFORMED,
                                )
                                for u in sorted(upstreams[table])
                            ]
                        ),
                    )
                )
    finally:
        con.close()
    return proposals


def _emitter(settings: Settings) -> Any:
    _models()
    from datahub.emitter.rest_emitter import DatahubRestEmitter

    return DatahubRestEmitter(gms_server=settings.datahub_gms_url, token=settings.datahub_gms_token)


def soft_delete(urns: list[str], settings: Settings) -> int:
    """Soft-delete entities (e.g. incident documents a live run wrote)."""
    if not urns:
        return 0
    models = _models()
    from datahub.emitter.mcp import MetadataChangeProposalWrapper as MCPW

    emitter = _emitter(settings)
    for urn in urns:
        emitter.emit_mcp(MCPW(entityUrn=urn, aspect=models.StatusClass(removed=True)))
    return len(urns)


def ingest_workspace(ws: Workspace, settings: Settings) -> int:
    emitter = _emitter(settings)
    proposals = build_proposals(ws.warehouse)
    for proposal in proposals:
        emitter.emit_mcp(proposal)
    return len(proposals)
