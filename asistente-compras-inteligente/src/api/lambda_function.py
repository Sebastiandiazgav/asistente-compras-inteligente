import json
import os
import boto3
import base64
import uuid
import time
import requests
import sys

from typing import TypedDict, List
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "tu-bucket-s3-por-defecto")
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "amazon.titan-text-express-v1")
POLLY_VOICE_ID = os.environ.get("POLLY_VOICE_ID", "Lupe")
AWS_REGION = os.environ.get("AWS_REGION", boto3.Session().region_name or "us-east-1")

s3_client = None
transcribe_client = None
bedrock_runtime_client = None
polly_client = None
llm = None
PRODUCT_DATABASE = None
agent_app = None

def initialize_aws_clients():
    global s3_client, transcribe_client, bedrock_runtime_client, polly_client, llm
    
    if s3_client is None:
        s3_client = boto3.client('s3', region_name=AWS_REGION)
    if transcribe_client is None:
        transcribe_client = boto3.client('transcribe', region_name=AWS_REGION)
    if bedrock_runtime_client is None:
        bedrock_runtime_client = boto3.client('bedrock-runtime', region_name=AWS_REGION)
    if polly_client is None:
        polly_client = boto3.client('polly', region_name=AWS_REGION)
    
    if llm is None and bedrock_runtime_client:
        try:
            llm = ChatBedrock(
                client=bedrock_runtime_client,
                model_id=BEDROCK_MODEL_ID,
                model_kwargs={"temperature": 0.1}
            )
        except Exception as e:
            print(f"Error inicializando LLM del Agente: {e}")
            raise e 
    print("Clientes AWS y LLM inicializados/verificados.")

class AgentState(TypedDict):
    userInput: str
    intent: str
    entities: dict
    catalogQueryResult: List[dict]
    finalResponse: str
    callLog: List[str]

def load_product_database_lambda():
    global PRODUCT_DATABASE
    if PRODUCT_DATABASE is not None:
        return PRODUCT_DATABASE

    db_path = "products.json" 
    if not os.path.exists(db_path):
        db_path = os.path.join("data", "products.json")

    try:
        with open(db_path, 'r', encoding='utf-8') as f:
            PRODUCT_DATABASE = json.load(f)
        print(f"Base de datos de productos (Lambda) cargada desde '{db_path}'")
        return PRODUCT_DATABASE
    except FileNotFoundError:
        print(f"LAMBDA_CATALOG_ERROR: Archivo DB no encontrado en '{db_path}'.")
        PRODUCT_DATABASE = []
        return []
    except Exception as e:
        print(f"LAMBDA_CATALOG_ERROR: Error cargando DB: {e}")
        PRODUCT_DATABASE = []
        return []

def interpret_user_input_lambda(state: AgentState):
    print("---LAMBDA AGENTE NODO: Interpretando Entrada---")
    user_input = state["userInput"]
    current_call_log = state.get("callLog", [])
    global llm
    if llm is None:
        print("Error crítico en interpret_user_input_lambda: LLM no está inicializado.")
        current_call_log.append("LAMBDA_NLU_ERROR: LLM no inicializado.")
        return {"intent": "error_nlu_llm", "entities": {}, "callLog": current_call_log}

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(
            content=(
                "Tu única función es analizar la consulta del usuario y DEVOLVER ÚNICAMENTE UN OBJETO JSON VÁLIDO. "
                "NO INCLUYAS NINGÚN TEXTO EXPLICATIVO, SALUDO, COMENTARIO, O CUALQUIER OTRA COSA FUERA DEL OBJETO JSON. "
                "El objeto JSON debe tener exactamente dos claves de nivel superior: 'intent' (un string con la intención del usuario) y 'entities' (un diccionario que contenga las entidades extraídas como pares clave-valor). "
                "Las intenciones posibles son: 'buscar_producto', 'comparar_productos', 'pedir_recomendacion', 'ver_carrito', 'saludar', 'despedirse', 'otra'. "
                "Las entidades comunes a extraer son: 'nombre_producto', 'marca', 'categoria', 'color', 'talla', 'precio_maximo', 'caracteristicas_adicionales'. "
                "Para la entidad 'categoria', si el usuario menciona un tipo de producto como 'botas de montaña', 'televisor LED', o 'zapatillas para correr', intenta extraer la categoría más específica posible (ej. 'botas de montaña', 'televisor LED', 'zapatillas para correr') o la categoría principal (ej. 'botas', 'televisor', 'zapatillas') como valor para la clave 'categoria' en el diccionario 'entities'. "
                "Para la entidad 'marca', si el usuario menciona un nombre específico junto al tipo de producto (ej. 'televisor SuperVision'), considera ese nombre como la marca. Extrae el nombre tal cual lo dice el usuario como valor para la clave 'marca' en 'entities'. "
                "Si una entidad no se encuentra, no incluyas su clave en el diccionario 'entities'. "
                "Si la intención no es clara o no hay entidades útiles, usa la intención 'otra' y un diccionario 'entities' vacío. "
                "La estructura general del JSON que debes devolver es: {\"intent\": \"valor_de_la_intencion\", \"entities\": {\"nombre_entidad_1\": \"valor_entidad_1\", \"nombre_entidad_2\": \"valor_entidad_2\"}}. "
                "REPITO: Tu respuesta DEBE SER SOLO EL OBJETO JSON y nada más."
            )
        ),
        HumanMessage(content=f"Analiza esta consulta: {user_input}")
    ])
    response_content = ""
    parsed_json_string = ""
    try:
        formatted_prompt = prompt_template.format_messages(userInput=user_input)
        ai_response = llm.invoke(formatted_prompt)
        response_content = ai_response.content.strip()
        print(f"Respuesta cruda del LLM para NLU (Agente-Titan, extracción mejorada): {response_content}")
        
        json_start_index = response_content.find('{')
        if json_start_index != -1:
            open_braces = 0
            for i, char in enumerate(response_content[json_start_index:]):
                if char == '{': open_braces += 1
                elif char == '}': open_braces -= 1
                if open_braces == 0 and char == '}':
                    parsed_json_string = response_content[json_start_index : json_start_index + i + 1]
                    print(f"String JSON extraído por contador de llaves: {parsed_json_string}")
                    break
            if not parsed_json_string:
                print("Contador de llaves no encontró un JSON balanceado.")
        
        if not parsed_json_string:
            print("No se pudo aislar un string JSON claro. Intentando parsear la respuesta cruda (puede fallar).")
            parsed_json_string = response_content

        parsed_response = json.loads(parsed_json_string)
        intent = parsed_response.get("intent", "otra")
        entities = parsed_response.get("entities", {})
        current_call_log.append(f"LAMBDA_NLU: Input='{user_input}', Intent='{intent}', Entities='{json.dumps(entities)}', ParsedJSON='{parsed_json_string[:100]}...' ,LLM_Raw_Output='{response_content[:100]}...'")
        print(f"Intención extraída (Agente, extracción mejorada): {intent}")
        print(f"Entidades extraídas (Agente, extracción mejorada): {entities}")
        return {"intent": intent, "entities": entities, "callLog": current_call_log}
    except json.JSONDecodeError as e:
        print(f"Error FINAL al decodificar la respuesta JSON del LLM (Agente-Titan, extracción mejorada): {e}")
        print(f"String que se intentó parsear como JSON: '{parsed_json_string}'")
        print(f"Respuesta cruda completa del LLM: '{response_content}'")
        current_call_log.append(f"LAMBDA_NLU_ERROR: JSONDecodeError. AttemptedParse='{parsed_json_string[:100]}...', LLM_Raw_Output='{response_content[:100]}...'")
        return {"intent": "error_nlu_format", "entities": {}, "callLog": current_call_log}
    except Exception as e:
        print(f"Error inesperado durante la NLU (Agente-Titan, extracción mejorada): {e}")
        current_call_log.append(f"LAMBDA_NLU_ERROR: Exception - {str(e)}")
        return {"intent": "error_nlu_unexpected", "entities": {}, "callLog": current_call_log}

def query_product_catalog_lambda(state: AgentState):
    print("---LAMBDA AGENTE NODO: Consultando Catálogo---")
    entities = state.get("entities", {})
    intent = state.get("intent", "")
    current_call_log = state.get("callLog", [])
    global PRODUCT_DATABASE
    if PRODUCT_DATABASE is None or not PRODUCT_DATABASE:
        print("Error: Base de datos de productos (Lambda) no disponible.")
        current_call_log.append("LAMBDA_CATALOG_ERROR: DB no disponible.")
        return {"catalogQueryResult": [], "callLog": current_call_log}

    if intent not in ["buscar_producto", "pedir_recomendacion", "comparar_productos"]:
        current_call_log.append(f"LAMBDA_CATALOG_SKIP: Intención '{intent}'.")
        return {"catalogQueryResult": [], "callLog": current_call_log}
    print(f"Consultando catálogo (Agente, Lógica Refinada) con entidades: {entities}")
    results = []
    if not entities:
        print("No se proporcionaron entidades específicas para filtrar el catálogo.")
        results = []
    else:
        for product in PRODUCT_DATABASE:
            match_score = 0; perfect_match_needed = 0
            entity_categoria = entities.get("categoria", "").lower()
            if entity_categoria:
                perfect_match_needed += 1
                product_categoria_actual = product.get("categoria", "").lower()
                if entity_categoria in product_categoria_actual or product_categoria_actual in entity_categoria: match_score += 1
                elif entity_categoria in product.get("nombre", "").lower(): match_score += 0.5
            entity_marca = entities.get("marca", "").lower()
            if entity_marca:
                perfect_match_needed += 1
                if entity_marca in product.get("marca", "").lower(): match_score += 1
            entity_nombre_producto = entities.get("nombre_producto", "").lower()
            if entity_nombre_producto:
                perfect_match_needed +=1
                if entity_nombre_producto in product.get("nombre", "").lower(): match_score += 1
            entity_tamano = entities.get("tamaño", "").lower()
            if entity_tamano:
                perfect_match_needed +=1
                tamano_numerico_entidad = "".join(filter(str.isdigit, entity_tamano))
                if tamano_numerico_entidad:
                    if tamano_numerico_entidad in product.get("nombre", "").lower() or \
                       any(tamano_numerico_entidad in str(car).lower() for car in product.get("caracteristicas", [])):
                        match_score += 1
            entity_color = entities.get("color", "").lower()
            if entity_color:
                perfect_match_needed +=1
                product_colores = [str(c).lower() for c in product.get("colores", [])]
                if product_colores and entity_color in product_colores: match_score += 1
            entity_talla = str(entities.get("talla", ""))
            if entity_talla:
                perfect_match_needed +=1
                product_tallas = [str(t) for t in product.get("tallas_disponibles", [])]
                if product_tallas and entity_talla in product_tallas: match_score += 1
            if perfect_match_needed > 0 and match_score >= perfect_match_needed:
                results.append(product)
    
    if not results:
        print("No se encontraron productos que coincidan exactamente con todas las entidades proporcionadas.")
        current_call_log.append(f"LAMBDA_CATALOG_QUERY: Entities='{json.dumps(entities)}', Result='No products found (strict match)'")
    else:
        print(f"Productos encontrados (Lógica Refinada): {len(results)}")
        summary_results = [{"id": p.get("id"), "nombre": p.get("nombre")} for p in results[:3]]
        current_call_log.append(f"LAMBDA_CATALOG_QUERY: Entities='{json.dumps(entities)}', Found='{len(results)} items', ExampleResults='{json.dumps(summary_results)}'")
    return {"catalogQueryResult": results, "callLog": current_call_log}

def generate_response_lambda(state: AgentState):
    print("---LAMBDA AGENTE NODO: Generando Respuesta (Prompt Fluidez v2)---")
    user_input = state.get("userInput", "")
    intent = state.get("intent", "")
    entities = state.get("entities", {})
    catalog_results = state.get("catalogQueryResult", [])
    current_call_log = state.get("callLog", [])
    global llm
    if llm is None:
        print("Error crítico en generate_response_lambda: LLM no está inicializado.")
        current_call_log.append("LAMBDA_RESPONSE_ERROR: LLM no inicializado.")
        return {"finalResponse": "Lo siento, estoy experimentando un problema técnico y no puedo generar una respuesta en este momento.", "callLog": current_call_log}

    context_for_llm = f"La consulta original del usuario (transcrita de voz) fue: '{user_input}'.\n"
    if intent: 
        context_for_llm += f"La intención identificada fue: '{intent}'.\n"
    if entities: 
        context_for_llm += f"Las entidades relevantes extraídas fueron: {json.dumps(entities)}.\n"

    if intent.startswith("error_nlu"):
        final_response_text = "Lo siento, tuve algunos problemas para entender completamente tu solicitud de voz. ¿Podrías intentar reformularla o ser un poco más específico?"
        current_call_log.append(f"LAMBDA_RESPONSE_GEN: Intent='{intent}', Generated_Response='{final_response_text}'")
        return {"finalResponse": final_response_text, "callLog": current_call_log}

    if intent in ["saludar", "despedirse"] and not catalog_results:
        if intent == "saludar":
            final_response_text = "¡Hola! Soy tu asistente de compras inteligente. ¿Cómo puedo ayudarte hoy?"
        else: 
            final_response_text = "¡Hasta pronto! Que tengas un excelente día."
        current_call_log.append(f"LAMBDA_RESPONSE_GEN: Intent='{intent}', Generated_Response='{final_response_text}'")
        return {"finalResponse": final_response_text, "callLog": current_call_log}

    system_message_content = ""
    human_message_instruction = (
        "Basándote en el contexto anterior, formula una respuesta ÚNICA, CONVERSACIONAL, AMIGABLE y DIRECTA para el usuario. "
        "NO simules múltiples turnos de conversación. NO uses prefijos como 'Bot:'. NO uses comillas innecesarias alrededor de toda tu respuesta. "
        "Simplemente proporciona la frase que el asistente debería decir."
    )

    if not catalog_results:
        if intent in ["buscar_producto", "pedir_recomendacion", "comparar_productos"]:
            context_for_llm += "No se encontraron productos en el catálogo que coincidan exactamente con la búsqueda del usuario.\n"
            system_message_content = (
                "Eres un asistente de compras virtual servicial y empático. "
                "Tu tarea es informar al usuario de manera clara y amigable que no se encontraron productos para su búsqueda. "
                "Anímale a intentar con diferentes términos, a ser más general, o pregunta si puedes ayudarle con otra cosa. "
            )
        else: 
            context_for_llm += "No se requirió consultar el catálogo para esta interacción.\n"
            system_message_content = (
                "Eres un asistente de compras virtual servicial y empático. "
                "Responde de forma concisa, amigable y directa a la consulta del usuario basándote en la intención y el contexto proporcionado. "
            )
    else: 
        context_for_llm += "Se encontraron los siguientes productos que podrían interesarle al usuario:\n"
        for i, product in enumerate(catalog_results[:2]): 
            product_info = f"  - {product.get('nombre', 'Nombre no disponible')}"
            if product.get('marca'): product_info += f" (Marca: {product.get('marca')})"
            if product.get('precio'): product_info += f", Precio: ${product.get('precio')}"
            context_for_llm += product_info + "\n"
        if len(catalog_results) > 2:
            context_for_llm += f"  ... y {len(catalog_results) - 2} producto(s) más similares.\n"
        
        system_message_content = (
            "Eres un asistente de compras virtual experto, amigable y servicial. "
            "Basándote en la consulta del usuario y los productos encontrados (listados en el contexto), "
            "genera una respuesta ÚNICA, FLUIDA, CONVERSACIONAL y DIRECTA. "
            "Presenta la información de manera clara y útil. Si hay productos, menciona uno o dos de los más relevantes. "
            "Puedes concluir preguntando si desea más detalles sobre alguno en particular, si quiere ver otras opciones, o si hay algo más en lo que puedas asistir. "
            "Ejemplo de una buena respuesta si encuentras 'Botas de Montaña TerraTrek': 'Encontré las Botas de Montaña TerraTrek. Son impermeables y cuestan $180.00. ¿Te gustaría saber más detalles sobre estas botas o prefieres que busque otras opciones?'"
        )
    
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_message_content),
        HumanMessage(content=context_for_llm + "\n" + human_message_instruction)
    ])
    
    final_response_text = "Lo siento, no pude generar una respuesta en este momento." 
    try:
        formatted_prompt = prompt_template.format_messages()
        ai_response = llm.invoke(formatted_prompt)
        cleaned_response = ai_response.content.strip()
        prefixes_to_remove = ["Bot:", "Respuesta:", "Respuesta del asistente:"]
        for prefix in prefixes_to_remove:
            if cleaned_response.lower().startswith(prefix.lower()):
                cleaned_response = cleaned_response[len(prefix):].strip()
        
        if (cleaned_response.startswith("'") and cleaned_response.endswith("'")) or \
           (cleaned_response.startswith('"') and cleaned_response.endswith('"')):
            cleaned_response = cleaned_response[1:-1]

        final_response_text = cleaned_response
        print(f"Respuesta generada y limpiada por LLM (Agente, fluidez v2): {final_response_text}")
        current_call_log.append(f"LAMBDA_RESPONSE_GEN: Intent='{intent}', Found='{len(catalog_results)}', Generated='{final_response_text[:100]}...'")
    except Exception as e:
        print(f"Error en generación de respuesta LLM (Agente, fluidez v2): {e}")
        current_call_log.append(f"LAMBDA_RESPONSE_GEN_ERROR: {str(e)}")
    
    return {"finalResponse": final_response_text, "callLog": current_call_log}

def compile_agent_graph():
    global agent_app
    if agent_app is not None:
        return agent_app
        
    workflow = StateGraph(AgentState)
    workflow.add_node("nlu_parser_lambda", interpret_user_input_lambda)
    workflow.add_node("catalog_tool_lambda", query_product_catalog_lambda)
    workflow.add_node("response_generator_lambda", generate_response_lambda)
    workflow.set_entry_point("nlu_parser_lambda")
    workflow.add_edge("nlu_parser_lambda", "catalog_tool_lambda")
    workflow.add_edge("catalog_tool_lambda", "response_generator_lambda")
    workflow.add_edge("response_generator_lambda", END)
    agent_app = workflow.compile()
    print("Grafo del Agente LangGraph compilado para Lambda.")
    return agent_app

def transcribe_audio_lambda(audio_bytes: bytes, input_audio_format: str = 'webm'):
    global s3_client, transcribe_client
    if not s3_client or not transcribe_client:
        raise Exception("Clientes S3 o Transcribe no inicializados.")

    unique_id = str(uuid.uuid4())
    s3_object_key = f"transcribe-input-lambda/{unique_id}.{input_audio_format}" 
    transcription_job_name = f"LambdaTranscription-{unique_id}"
    temp_audio_path = f"/tmp/{unique_id}.{input_audio_format}"

    try:
        with open(temp_audio_path, 'wb') as f:
            f.write(audio_bytes)
        
        s3_client.upload_file(temp_audio_path, S3_BUCKET_NAME, s3_object_key)
        media_file_uri = f"s3://{S3_BUCKET_NAME}/{s3_object_key}"
        
        transcribe_client.start_transcription_job(
            TranscriptionJobName=transcription_job_name,
            Media={'MediaFileUri': media_file_uri},
            MediaFormat=input_audio_format,
            LanguageCode='es-US'
        )
        while True:
            status = transcribe_client.get_transcription_job(TranscriptionJobName=transcription_job_name)
            job_status = status['TranscriptionJob']['TranscriptionJobStatus']
            if job_status in ['COMPLETED', 'FAILED']: break
            time.sleep(5)
        
        if job_status == 'FAILED':
            raise Exception(f"Trabajo de transcripción falló: {status['TranscriptionJob'].get('FailureReason')}")
        
        transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
        response = requests.get(transcript_uri)
        response.raise_for_status()
        transcript_json = response.json()
        return transcript_json['results']['transcripts'][0]['transcript']
    finally:
        if os.path.exists(temp_audio_path): os.remove(temp_audio_path)
        try: s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=s3_object_key)
        except Exception: pass

def synthesize_speech_lambda(text_to_synthesize: str):
    global polly_client
    if not polly_client:
        raise Exception("Cliente Polly no inicializado.")
    if not text_to_synthesize: return None

    response = polly_client.synthesize_speech(
        Text=text_to_synthesize,
        OutputFormat='mp3',
        VoiceId=POLLY_VOICE_ID,
        Engine='standard'
    )
    if 'AudioStream' in response:
        return response['AudioStream'].read()
    return None

def lambda_handler(event, context):
    print(f"Evento recibido: {json.dumps(event)}")
    global agent_app 
    try:
        initialize_aws_clients()
        load_product_database_lambda()
        if agent_app is None:
            compile_agent_graph()
        
        if 'body' not in event:
            return {'statusCode': 400, 'body': json.dumps({'error': 'Cuerpo de la solicitud no encontrado.'})}
        body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        if 'audio_base64' not in body:
            return {'statusCode': 400, 'body': json.dumps({'error': 'audio_base64 no encontrado en el cuerpo.'})}
        
        audio_base64_string = body['audio_base64']
        input_audio_format = body.get('audio_format', 'webm') 
        audio_bytes = base64.b64decode(audio_base64_string)
        print(f"Audio decodificado, {len(audio_bytes)} bytes, formato: {input_audio_format}")

        transcribed_text = transcribe_audio_lambda(audio_bytes, input_audio_format)
        if not transcribed_text:
            raise Exception("La transcripción falló o devolvió texto vacío.")
        print(f"Texto Transcrito: {transcribed_text}")

        if agent_app is None:
            raise Exception("El grafo del agente no se pudo compilar.")

        agent_initial_input = {"userInput": transcribed_text, "callLog": []}
        agent_final_state = agent_app.invoke(agent_initial_input)
        agent_response_text = agent_final_state.get('finalResponse', "No pude generar una respuesta.")
        print(f"Respuesta del Agente (Texto): {agent_response_text}")

        response_audio_bytes = synthesize_speech_lambda(agent_response_text)
        response_audio_base64 = None
        if response_audio_bytes:
            response_audio_base64 = base64.b64encode(response_audio_bytes).decode('utf-8')
            print("Respuesta de audio sintetizada y codificada en base64.")
        else:
            print("No se pudo sintetizar el audio de respuesta.")

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*' 
            },
            'body': json.dumps({
                'inputText': transcribed_text,
                'agentResponseText': agent_response_text,
                'agentResponseAudioBase64': response_audio_base64,
                'agentCallLog': agent_final_state.get('callLog', [])
            })
        }
    except Exception as e:
        print(f"Error en lambda_handler: {e}")
        import traceback
        traceback.print_exc()
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({'error': str(e)})
        }

if __name__ == '__main__':
    print("Ejecutando prueba local de lambda_handler (requiere un archivo de audio base64)...")
    print("Para probar localmente, descomenta y configura el mock_event con un audio base64.")
