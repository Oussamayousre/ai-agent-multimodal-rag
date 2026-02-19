import os

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric,FaithfulnessMetric,ContextualPrecisionMetric,ContextualRelevancyMetric
# Replace this with the actual output from your LLM application
actual_output = "We offer a 30-day full refund at no extra cost."
# Replace this with the expected output of your RAG generator
# expected_output = "You are eligible for a 30 day full refund at no extra cost."
# # Replace this with the actual retrieved context from your RAG pipeline
# retrieval_context = ["All customers are eligible for a 30 day full refund at no extra cost."]
class Rag_Eval() : 

    def __init__(self):
        AnswerRelevancy = AnswerRelevancyMetric(
                threshold=0.7,
                model="gpt-4.1",
                include_reason=True)
        FaithfulnessMetric  = FaithfulnessMetric(
                threshold=0.7,
                model="gpt-4.1",
                include_reason=True)
        FaithfulnessMetric = ContextualRelevancyMetric(
                threshold=0.7,
                model="gpt-4.1",
                include_reason=True)
        eval_dict = { 
            "AnswerRelevancy" : AnswerRelevancy,
            "FaithfulnessMetric" : FaithfulnessMetric,
            "FaithfulnessMetric" : FaithfulnessMetric
        }
    
    def Eval_rag(self,metrics_list,input,actual_output,retrieval_context,expected_output = "") : 
        test_case = LLMTestCase(
                input=input,
                actual_output=actual_output,
                expected_output=expected_output,
                retrieval_context=retrieval_context
            )
        
        evaluate(test_cases=[test_case], metrics=[metric for metric in metrics_list])
        # print(eval_output)

# To run metric as a standalone
# metric.measure(test_case)
# print(metric.score, metric.reason)

