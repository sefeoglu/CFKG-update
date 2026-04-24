from rdflib import Graph, Namespace
from rdflib.namespace import RDF
import json # Import the json module

class SHACLValidation:
    def __init__(self, shapes_file):
        self.shapes_file = shapes_file
        self.graph = Graph()
        self.graph.parse(shapes_file, format="turtle")
        constraints = self.extract_constraints()
        self.constraints = constraints
        

    def extract_constraints(self):
        
        constraints = []
        SH = Namespace("http://www.w3.org/ns/shacl#")
        for shape in self.graph.subjects(RDF.type, SH.NodeShape):
            constraint = {}

            # Relation trigger
            target_relation = self.graph.value(shape,   SH.targetSubjectsOf)
            if target_relation:
                constraint["targetSubjectsOf"] = str(target_relation)

            # Severity
            severity = self.graph.value(shape, SH.severity)
            if severity:
                constraint["severity"] = str(severity)

            # SPARQL constraint
            sparql_node = self.graph.value(shape, SH.sparql)
            if sparql_node:
                message = self.graph.value(sparql_node, SH.message)
                select = self.graph.value(sparql_node, SH.select)

                constraint["sparql"] = {
                    "message": str(message) if message else None,
                    "select": str(select) if select else None
                }

            if constraint:
                constraint["shape"] = str(shape)
                constraints.append(constraint)

        return constraints
    
    def check_constraints(self, detected_prediction):

        for c in self.constraints:
        
            if detected_prediction == json.loads(c['sparql']['message'])['detected']:
          
                return json.loads(c['sparql']['message'])['suggest']
        return None

