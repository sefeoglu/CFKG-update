
import argparse

class Incremental_KG_RE(object):
    def __init__(self, input_path="data.json", output_path="output.json"):
        self.input_path = input_path
        self.output_path = output_path

    def incremental_train(self):
        pass
    def uncertainty_estimation(self):
        pass
    def canonical_relation_detection(self):
        pass

    def llm_as_a_judge(self):
        pass
    def save_results(self):
        pass
    def pipeline(self):
        pass
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Incremental Knowledge Graph Relation Extraction")
    parser.add_argument("--data_path", type=str, default="data.json", help="Path to the input data file")
    parser.add_argument("--output_path", type=str, default="output.json", help="Path to the output data file")
    parser.add_argument("--model_path", type=str, default="model/", help="Path to save/load the model")
    parser.add_argument("--uncertainty_threshold", type=float, default=0.5, help="Threshold for uncertainty estimation")
    parser.add_argument("--llm_model", type=str, default="gemma", help="LLM model to use as a judge")
    parser.add_argument("--judge_model", type=str, default="gemma", help="LLM model to use as a judge")
    parser.add_argument("--uncertainty_method", type=str, default="aleatoric", help="Uncertainty estimation method")
    parser.add_argument('--synaptic_method', type=str, default='ewc', help='Synaptic method for incremental learning')

    args = parser.parse_args()

    print("Starting Incremental Knowledge Graph Relation Extraction with Uncertainty Estimation...")
    
    model = Incremental_KG_RE()
    model.incremental_train()
    model.uncertainty_estimation()
    model.canonical_relation_detection()
    model.llm_as_a_judge()
    print("Process Completed.")
