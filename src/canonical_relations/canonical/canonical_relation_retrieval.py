# imports
import os
import json
import random 
import argparse
from nltk.corpus import wordnet as wn

# pip install nltk
import json
from collections import defaultdict
import nltk
from nltk.corpus import wordnet as wn

# Ensure resources are present (no-op if already downloaded)
try:
    wn.ensure_loaded()
except:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

def syn_name(s):
    return f"{s.name()} :: {s.definition()}"

def lemma_names(synsets):
    out = set()
    for s in synsets:
        out.update(s.lemma_names())
    return sorted(out)

def unique_syn_names(synsets):
    return sorted({syn_name(s) for s in synsets})

def lemma_antonyms(s):
    ants = set()
    for l in s.lemmas():
        ants.update([a.name() for a in l.antonyms()])
    return sorted(ants)

def derivational_map(s):
    """Map lemma -> list of derivationally related lemma names."""
    dmap = defaultdict(set)
    for l in s.lemmas():
        rel = l.derivationally_related_forms()
        for r in rel:
            dmap[l.name()].add(r.name())
    return {k: sorted(v) for k, v in dmap.items() if v}

def coordinate_terms(s):
    """Siblings under any hypernym."""
    coords = set()
    for h in s.hypernyms() + s.instance_hypernyms():
        for sib in h.hyponyms() + h.instance_hyponyms():
            if sib != s:
                coords.update(sib.lemma_names())
    return sorted(coords)

def troponyms_via_verb_groups(s):
    """
    WordNet “troponyms” are manner relations between verbs.
    NLTK provides verb_groups() which often captures close verb relations (synonyms/troponyms).
    """
    out = set()
    if s.pos() == 'v':
        for g in s.verb_groups():
            if g != s:
                out.update(g.lemma_names())
    return sorted(out)

def collect_relations_for_synset(s):
    data = {
        "synset": s.name(),
        "pos": s.pos(),
        "definition": s.definition(),
        "examples": s.examples(),
        "lemmas": sorted(s.lemma_names()),

        # core binary relations
        "antonyms": lemma_antonyms(s),

        # taxonomy
        "hypernyms": unique_syn_names(s.hypernyms()),
        "instance_hypernyms": unique_syn_names(s.instance_hypernyms()),
        "hyponyms": unique_syn_names(s.hyponyms()),
        "instance_hyponyms": unique_syn_names(s.instance_hyponyms()),

        # meronymy/holonymy (3 flavors)
        "part_meronyms": unique_syn_names(s.part_meronyms()),
        "member_meronyms": unique_syn_names(s.member_meronyms()),
        "substance_meronyms": unique_syn_names(s.substance_meronyms()),
        "part_holonyms": unique_syn_names(s.part_holonyms()),
        "member_holonyms": unique_syn_names(s.member_holonyms()),
        "substance_holonyms": unique_syn_names(s.substance_holonyms()),

        # adjectives/adverbs extras
        "similar_tos": unique_syn_names(s.similar_tos()),
        "also_sees": unique_syn_names(s.also_sees()),
        "attributes": unique_syn_names(s.attributes()),

        # verbs: semantic inference
        "entailments": unique_syn_names(s.entailments()),
        "causes": unique_syn_names(s.causes()),
        "verb_groups": [g.name() for g in s.verb_groups()] if s.pos() == 'v' else [],

        # domains (only direction WordNet supports now)
        "topic_domains": unique_syn_names(s.topic_domains()),
        "region_domains": unique_syn_names(s.region_domains()),
        "usage_domains": unique_syn_names(s.usage_domains()),

        # backward domain links no longer in API
        "topic_members": [],
        "region_members": [],
        "usage_members": [],

        # cross-POS morphology
        "derivationally_related": derivational_map(s),

        # siblings
        # "coordinate_terms": coordinate_terms(s),
    }

    # add troponyms-like neighbors for verbs
    if s.pos() == 'v':
        data["troponyms_like"] = troponyms_via_verb_groups(s)

    return data

def wordnet_canonical_relations(word, pos=None, max_synsets=10):
    """
    word: string, e.g., 'car'
    pos: one of {'n','v','a','s','r'} or None for all POS
    Returns: dict with entries per synset.
    """
    synsets = wn.synsets(word, pos=pos)
    result = {
        "query": {"word": word, "pos": pos},
        "synset_count": len(synsets),
        "synsets": []
    }
    for s in synsets[:max_synsets]:
        result["synsets"].append(collect_relations_for_synset(s))
    return result

def pretty_print_relations(relations_dict, max_list=8):
    def shorten(lst):
        return lst if len(lst) <= max_list else lst[:max_list] + ["…"]
    print(f"Query: {relations_dict['query']}")
    print(f"Synsets found: {relations_dict['synset_count']}\n")
    for i, s in enumerate(relations_dict["synsets"], 1):
        print(f"[{i}] {s['synset']} :: {s['definition']}")
        print("  Lemmas:", ", ".join(shorten(s["lemmas"])))
        if s["antonyms"]:
            print("  Antonyms:", ", ".join(shorten(s["antonyms"])))
        # taxonomy
        if s["hypernyms"]:
            print("  Hypernyms:", ", ".join(shorten(s["hypernyms"])))
        if s["hyponyms"]:
            print("  Hyponyms:", ", ".join(shorten(s["hyponyms"])))
        # meronymy/holonymy
        for k in ["part_meronyms","member_meronyms","substance_meronyms","part_holonyms","member_holonyms","substance_holonyms"]:
            if s[k]:
                label = k.replace("_", " ").title()
                print(f"  {label}:", ", ".join(shorten(s[k])))
        # verbs
        if s["entailments"]:
            print("  Entailments:", ", ".join(shorten(s["entailments"])))
        if s["causes"]:
            print("  Causes:", ", ".join(shorten(s["causes"])))
        if s.get("troponyms_like"):
            print("  Troponyms-like:", ", ".join(shorten(s["troponyms_like"])))
        # adjectives etc.
        if s["similar_tos"]:
            print("  Similar-to:", ", ".join(shorten(s["similar_tos"])))
        if s["also_sees"]:
            print("  Also-see:", ", ".join(shorten(s["also_sees"])))
        if s["attributes"]:
            print("  Attributes:", ", ".join(shorten(s["attributes"])))
        # domains
        for k in ["topic_domains","region_domains","usage_domains"]:
            if s[k]:
                label = k.replace("_", " ").title()
                print(f"  {label}:", ", ".join(shorten(s[k])))
        # coordinates
        # if s["coordinate_terms"]:
        #     print("  Coordinate terms:", ", ".join(shorten(s["coordinate_terms"])))
        # derivational map
        if s["derivationally_related"]:
            print("  Derivationally related:")
            for base, rels in list(s["derivationally_related"].items())[:max_list]:
                more = "" if len(rels) <= max_list else " …"
                print("    -", base, "→", ", ".join(rels[:max_list]) + more)
        print()



def read_json(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def write_json(data, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)


def wordnet_canonical_relation_retrieval(relation):
    canonical =[]
    synsets = wn.synsets(relation)
    for s in synsets:
        print(s.name(), s.lemma_names())
        canonical.extend(s.lemma_names())
    canonical = list(set(canonical))
    return canonical

def run_wikidata_altlabel_retrieval(QID):
    pass
def wikidata_altlabel_retrieval(QID):

    pass
def clean_relation_prefix(relation):
    return relation.split(":")[-1].replace("_", " ").lower()

def get_all_canonical_relations(relation_path, type_,  output_path):


    relation_data = read_json(relation_path)
    print(relation_data[0].keys())
    ### 1. WordNet canonical relation retrieval

    canonical_results = []

    for key, relation_values in relation_data[0].items():

        print(relation_values)
        for variant in relation_values:
            relation_clean = clean_relation_prefix(variant)
            results= wordnet_canonical_relation_retrieval(relation_clean)
            # pretty_print_relations(results)

            canonicals = {"original_relation": relation_clean, "wordnet_canonicals": results}
     

        if type_ == "FewRel":
            QID = relation_values[0]['wikidata_qid']
            canonicals['wordnet_canonicals'] = results

        if type_ == "FewRel":
            QID = relation_values[0]['wikidata_qid']
            canonicals['wikidata_altlabels'] = run_wikidata_altlabel_retrieval(QID)

        results = {'relation': key, 'canonical_relations': canonicals}
        canonical_results.append(results)
    write_json(canonical_results, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--relation_path",
        type=str,
        default="/Users/sefika/phd_projects/llm-catastrophic-re/dataset/tacred/id2rel_tacred_normalized.json",
        help="Path to the relation list JSON file.",
        
    )
    parser.add_argument(
        "--output_path",    
        type=str,
        default="./tacred/canonical_relations/canonical_relations.json",
        help="Path to save the canonical relations JSON file.",
    )
    parser.add_argument(
        "--type_",
        type=str,
        default="TACRED",
        help="Dataset type for specific retrieval methods.",
    )
    args = parser.parse_args()

    get_all_canonical_relations(
        relation_path=args.relation_path,
        type_=args.type_,
        output_path=args.output_path,
    )

if __name__ == "__main__":
    main()