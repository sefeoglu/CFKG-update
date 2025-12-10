## Continual Factual Knowledge Graph Updates

<p align="center">
  <img src="https://github.com/sefeoglu/llm-catastrophic-re/blob/master/figs/method.png" width="500"/>
</p>
## Hallucination Taxonomy (ASCII Diagram)

```text
                             +--------------------------------------+
                             |      Undefined Relation Prediction   |
                             |        (Hallucination Umbrella)      |
                             +--------------------------+-----------+
                                                        |
                                                        v
    +----------------------------+-----------------------+--------------------------+-----------------------------+
    |                            |                       |                          |                             |
    v                            v                       v                          v
+-------------------+   +-------------------------+   +-------------------+   +------------------------------+
|  Similar Context  |   |  Part of Ground-Truth   |   |    New Relation   |   |  Fine-Grained Ground-Truth  |
|   Hallucination   |   |     Hallucination       |   |    Hallucination  |   |       Hallucination         |
+-------------------+   +-------------------------+   +-------------------+   +------------------------------+
| Prediction is     |   | Prediction is a         |   | Relation is       |   | Prediction is a more        |
| semantically      |   | component/generalized   |   | entirely new and  |   | specific version of the     |
| close to the      |   | subset of gold label.   |   | unrelated to GT.  |   | gold relation.              |
| gold label.       |   |                         |   |                   |   |                             |
|                   |   |                         |   | Examples:         |   | Example:                    |
| Example:          |   | Example:                |   |  - Japanese       |   |  child → son/daughter       |
| head of gov. →    |   | screenwriter → author   |   |    cargoer        |   |  sibling → sister/brother   |
| leader            |   |                         |   |  - Christmas      |   |                             |
|                   |   |                         |   |  - rail           |   |                             |
+-------------------+   +-------------------------+   +-------------------+   +------------------------------+

```
LLM with Catastrophic Forgetting approaches:
* SI : Synaptic Intelligence
* EWC : Elastic Weight Consolidation
* MAS : Memory-Aware Synapsis

## Evaluation
* Forgetting Measure
* BWT
* FWT
* BERTScore
* Mutual Information
* Loose Accuracy with Canonical Facts

## Folder Structure

## References
