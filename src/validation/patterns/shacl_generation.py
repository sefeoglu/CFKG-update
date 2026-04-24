import json, re
from pathlib import Path

src = Path("/mnt/data/canonicals.json")
data = json.loads(src.read_text(encoding="utf-8"))

def slug(label: str) -> str:
    x = (label or "").strip().lower()
    x = re.sub(r"[^a-z0-9]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    if not x:
        x = "rel"
    if x[0].isdigit():
        x = f"r_{x}"
    return x

PREFIX_BLOCK = (
    "@prefix sh:  <http://www.w3.org/ns/shacl#> .\n"
    "@prefix :    <http://example.org/family/> .\n\n"
)

SPARQL_PREFIXES = "PREFIX : <http://example.org/family/>\n"

def emit_suggest_shape(alt_label: str, canonical_label: str) -> str:
    alt_local = slug(alt_label)
    can_local = slug(canonical_label)
    alt_iri = f":{alt_local}"
    can_local_str = can_local  # used in JSON suggest list

    shape_name = f":CanonicalSuggest_{alt_local}_to_{can_local}"
    msg = (
        "{"
        f"\"type\":\"SUGGEST_RELATIONS\","
        f"\"detected\":\"{alt_label}\","
        f"\"canonical\":\"{canonical_label}\","
        f"\"suggest\":[\"{can_local_str}\"]"
        "}"
    )

    select = (
        SPARQL_PREFIXES +
        "SELECT $this ?path ?value WHERE {\n"
        f"  $this {alt_iri} ?value .\n"
        f"  BIND({alt_iri} AS ?path)\n"
        "}\n"
    )

    ttl = (
        "################################################\n"
        f"# Suggest canonical: \"{alt_label}\" -> \"{canonical_label}\"\n"
        "################################################\n"
        f"{shape_name} a sh:NodeShape ;\n"
        f"  sh:targetSubjectsOf {alt_iri} ;\n"
        "  sh:severity sh:Info ;\n"
        "  sh:sparql [\n"
        f"    sh:message \"{msg}\" ;\n"
        "    sh:select \"\"\"\n"
        f"{select}"
        "    \"\"\" ;\n"
        "  ] .\n\n"
    )
    return ttl

ttl_parts = [PREFIX_BLOCK]
shape_count = 0
pair_count = 0

for pid, rec in data.items():
    canonical = (rec.get("label") or "").strip()
    if not canonical:
        continue
    alts = rec.get("alt_labels") or []
    alts = [a.strip() for a in alts if isinstance(a, str) and a.strip()]

    # de-dup case-insensitively and drop canonical if it appears in alts
    seen = set()
    alts_unique = []
    for a in alts:
        k = a.lower()
        if k == canonical.lower():
            continue
        if k in seen:
            continue
        seen.add(k)
        alts_unique.append(a)

    pair_count += len(alts_unique)
    for alt in alts_unique:
        ttl_parts.append(emit_suggest_shape(alt, canonical))
        shape_count += 1

out_path = Path("/mnt/data/shapes.generated.ttl")
out_path.write_text("".join(ttl_parts), encoding="utf-8")

# show a small preview and a sanity check on spouse (P26)
preview = "".join(ttl_parts[:3])
spouse_info = (data.get("P26", {}).get("label"), (data.get("P26", {}).get("alt_labels") or [])[:8])
shape_count, pair_count, str(out_path), spouse_info, preview[:800]

