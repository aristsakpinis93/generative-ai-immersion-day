import json
import os
from langchain.chains import ConversationalRetrievalChain
from langchain import SagemakerEndpoint
from langchain.prompts.prompt import PromptTemplate
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from langchain.llms.sagemaker_endpoint import ContentHandlerBase, LLMContentHandler
from langchain.memory import ConversationBufferWindowMemory
from langchain import PromptTemplate, LLMChain
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory
from kendra.kendra_index_retriever import KendraIndexRetriever



REGION = os.environ.get('REGION')
KENDRA_INDEX_ID = os.environ.get('KENDRA_INDEX_ID')


# Generative LLM 

# Content Handler for Option 1 - FLAN-T5-XXL - please uncomment below if you used this option
# class ContentHandler(LLMContentHandler):
#     content_type = "application/json"
#     accepts = "application/json"

#     def transform_input(self, prompt, model_kwargs):
#         input_str = json.dumps({"text_inputs": prompt, "temperature": 0, "max_length": 200})
#         return input_str.encode('utf-8')
    
#     def transform_output(self, output):
#         response_json = json.loads(output.read().decode("utf-8"))
#         return response_json["generated_texts"][0]

# Content Handler for Option 2 - Falcon40b-instruct - please uncomment below if you used this option
# class ContentHandler(LLMContentHandler):
#     content_type = "application/json"
#     accepts = "application/json"

#     def transform_input(self, prompt, model_kwargs):
#         input_str = json.dumps({"inputs": prompt, "parameters": {"do_sample": False, "repetition_penalty": 1.1, "return_full_text": False, "max_new_tokens":100}})
#         return input_str.encode('utf-8')
    
#     def transform_output(self, output):
#         response_json = json.loads(output.read().decode("utf-8"))
#         return response_json[0]["generated_text"]


content_handler = ContentHandler()

llm=SagemakerEndpoint(
    endpoint_name="***ENDPOINT_NAME***",
    region_name=REGION, 
    content_handler=content_handler, 
)

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language. 

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""


CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)


def lambda_handler(event, context):
    print(event)
    body = json.loads(event['body'])
    print(body)
    query = body['query']
    uuid = body['uuid']
    print(query)
    print(uuid)

    message_history = DynamoDBChatMessageHistory(table_name="MemoryTable", session_id=uuid)
    memory = ConversationBufferWindowMemory(memory_key="chat_history", chat_memory=message_history, return_messages=True, k=3)

    retriever = KendraIndexRetriever(kendraindex=KENDRA_INDEX_ID, 
      awsregion=REGION, 
      return_source_documents=True)
    qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, condense_question_prompt=CONDENSE_QUESTION_PROMPT, verbose=True)


    response = qa.run(query)   

    return {
            'statusCode': 200,
            'body': json.dumps(response)
        }
