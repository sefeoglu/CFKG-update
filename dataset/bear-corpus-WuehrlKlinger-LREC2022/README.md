# Recovering Patient Journeys: A Corpus of Biomedical Entities and Relations on Twitter (BEAR)
This zip file includes supplementary material for the paper Recovering Patient Journeys: A Corpus of Biomedical Entities and Relations on Twitter (BEAR) by Amelie Wuehrl and Roman
Klinger, accepted at LREC 2022.

Please cite this paper as:

```
@InProceedings{wuehrl_klinger_2022,
  author    = {Wuehrl, Amelie  and  Klinger, Roman},
  title     = {Recovering Patient Journeys: A Corpus of Biomedical Entities and Relations on  Twitter (BEAR)},
  booktitle      = {Proceedings of The 13th Language Resources and Evaluation Conference},
  month          = {June},
  year           = {2022},
  address        = {Marseille, France},
  publisher      = {European Language Resources Association}
}
```

# Content
This zip file contains the following the following files and folder:

* `README.md`: this file
* `supplementary-material/annotation-guidelines.pdf`: the document that has been developed
  during the annotation process. It documents the annotation task of
  the annotators.
* `supplementary-material/query-terms.txt`: The terms that we used to collect
  the data via the Twitter API.
* `supplementary-material/relational-terms.txt`: The relational terms that we use to filter the tweets.
* `corpus/` - the annotated corpus (described below). We provide annotations from both annotators (A1, A2) and an aggregated version which combines both annotations. This directory
  contains the actual tweet texts which is in line with the Twitter
  terms and conditions due to its limited size (see the
  [Twitter terms](https://developer.twitter.com/en/developer-terms/agreement-and-policy))


# Corpus Data

We provide an adjudicated version of the BEAR dataset (bear.jsonl). Additionally, we provide a file with the individual annotations from both annotators (a1.jsonl, a2.jsonl). Each line in those files is a json string with the following keys:

* `doc_id`: anonymized Twitter ID
* `annotator`: abbreviation for the annotator
* `doc_text`: string with tweet text. Note that @mentions have been replaced with a generic @username to anonymize the data.
* `entities`: optional, if one or more entities were annotated this contains a dictionary in the entity annotation. 
    * Dictionary keys are identifiers for the entity, `tag` refers to the entity class, `begin` and `end` refer to the character on and offset of the entity in the `doc_text`. 
    * An example entry: \{`347`: \{`tag`: `treat_drug`, `begin`: 24, `end`: 31, `char_sequence`: "aspirin"\}
* `relations`: a list of dictionaries containing annotated relations. 
    * Example entry: \{`rel_tag`: `treats`, `start_entity`: `437`, `end_entity`: `455`\}. `rel_tag` refers to the relation class, `start_entity` and `end_entity` reference the ids of the connected entities.



# License
The annotations in this dataset are licensed under a CC BY-SA license: https://creativecommons.org/licenses/by-sa/4.0/