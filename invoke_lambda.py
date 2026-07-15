import json
import boto3

sagemaker_runtime = boto3.client('sagemaker-runtime', region_name="eu-central-1")

system_prompt = """You are a helpful and empathetic customer service assistant.
Do not introduce yourself with any name.
Always be polite, professional, and solution-focused.
Produce one answer only.
Do not continue the conversation.
Do not generate another system, user, or assistant turn.
Do not invent placeholders, names, phone numbers, or links.
Keep the answer short and action-focused."""

def lambda_handler(event, context):
    endpoint_name = "llama3endpoint--v1"
    body = json.loads(event["body"])   
    prompt =  body['prompt']  # get prompt from event

    input_text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    payload = {
        "inputs": input_text,
        "parameters": {
            "do_sample":          False,
            "top_p":              0.7,
            "temperature":        0.2,
            "top_k":              40,
            "max_new_tokens":     150,
            "return_full_text":   False,
            "repetition_penalty": 1.12,
            "stop": [
                "<|eot_id|>",
                "<|start_header_id|>system<|end_header_id|>",
                "<|start_header_id|>user<|end_header_id|>",
                "<|start_header_id|>assistant<|end_header_id|>",
            ],
        },
    }

    try:
        response      = sagemaker_runtime.invoke_endpoint(
            EndpointName = endpoint_name,
            ContentType  = "application/json",
            Body         = json.dumps(payload)
        )
        response_body  = json.loads(response["Body"].read().decode())  
        generated_text = response_body[0]["generated_text"]

        return {
            "statusCode": 200,                           # 
            "body": json.dumps({
                "generated_text": generated_text
            })
        }

    except Exception as e:
        print(f"error happened: {e}")                    # 
        return {
            "statusCode": 500,                           # 
            "body": json.dumps({
                "error": str(e)
            })
        }