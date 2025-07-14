# Reload the new hallucinated relation list from the new user dataset
import pandas as pd
import ast
from collections import Counter
from sklearn.preprocessing import FunctionTransformer
data_2 = pd.read_csv("/Users/sefika/phd_projects/llm-catastrophic-re/results/si_tacred_flan_random_results.csv")['hallucination'].tolist()

def normalize_relations(halluc_list):
    results = []
    for entry in halluc_list:
        try:
            parsed = ast.literal_eval(entry)
        except:
            parsed = []
        cleaned = []
        for rel in parsed:
            rel = rel.strip()
            if ':' in rel:
                rel = rel.split(':', 1)[1]
            rel = rel.replace("provinces", "province").replace("countries", "country").replace("cities", "city").replace("members", "member").replace("employees", "employee").replace("affiliation", "religion").replace("headquarters", "headquarter").replace("alternate_names", "alternate_name").replace("religious_affiliation", "religion")
            rel = rel.replace("dates", "date")  # plural to singular
            cleaned.append(rel)
        results.extend(cleaned)
    return results
# Normalize and clean
normalized_relations_2 = normalize_relations(data_2["hallucination"])
relation_counts_2 = Counter(normalized_relations_2)
df_hallucinated_2 = pd.DataFrame(relation_counts_2.items(), columns=["relation_core", "count"])
df_hallucinated_2["is_in_tacred"] = df_hallucinated_2["relation_core"].apply(lambda x: x in tacred_valid_cores)

# Filter those not in TACRED
df_not_in_tacred_2 = df_hallucinated_2[df_hallucinated_2["is_in_tacred"] == False].reset_index(drop=True)

# tools.display_dataframe_to_user(name="Hallucinated Relations Not in TACRED (Dataset 2)", dataframe=df_not_in_tacred_2)
