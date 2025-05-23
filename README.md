# Prototipo de Voicebot con Capacidades de Agente

Este proyecto es un prototipo funcional de un agente inteligente controlado por voz, desarrollado como parte del "Desafío: Prototipo de Voicebot con Capacidades de Agente". El objetivo principal es demostrar la capacidad de interpretar instrucciones habladas, procesarlas internamente mediante un agente inteligente y generar una respuesta coherente tanto en texto como en voz.

El caso de uso implementado es un **Asistente de Compras Inteligente**, capaz de entender consultas sobre productos, buscar en un catálogo simulado y proporcionar información relevante al usuario.


## Arquitectura General del Prototipo (Nivel 3)

El sistema se compone de tres partes principales:

1.  **Frontend (UI de Voz):** Una página web simple  que permite al usuario grabar su consulta de voz usando el micrófono del navegador.
2.  **Backend (API y Lógica del Agente):**
    * **Amazon API Gateway (HTTP API):** Expone un endpoint HTTP que la UI consume.
    * **AWS Lambda (`lambda_function.py`):** El cerebro del sistema. Orquesta los siguientes servicios:
        * **Amazon Transcribe:** Convierte el audio de voz del usuario (recibido desde la UI) a texto.
        * **Agente Inteligente (LangGraph + Amazon Bedrock):**
            * Procesa el texto transcrito para NLU (Extracción de Intención y Entidades) usando un modelo fundacional de Amazon Bedrock (ej. Titan Text Express).
            * Consulta un catálogo de productos simulado (desde `products.json`).
            * Genera una respuesta textual coherente usando Amazon Bedrock.
        * **Amazon Polly:** Convierte la respuesta textual del agente a audio (voz).
    * La Lambda devuelve la respuesta textual y el audio (codificado en base64) a la UI.
3.  **Datos:**
    * `products.json`: Un archivo JSON que simula un catálogo de productos, utilizado por el agente.

## Estructura del Repositorio


asistente-compras-inteligente/
│
├── .github/                # (Opcional) Workflows de GitHub Actions
├── .vscode/                # (Opcional) Configuración de VS Code
├── data/                   # Archivos de datos
│   ├── products.json       # Catálogo de productos simulado
│   └── (audios de prueba/respuesta generados)
├── docs/                   # Documentación adicional y diagramas
│   └── (diagramas de arquitectura, funcional, etc.)
├── frontend/               # Interfaz de usuario
│   └── voice_ui.html       # UI para captura de voz e interacción
├── lambda_layer_stuff/     # Archivos para crear la capa de Lambda (instrucciones abajo)
│   └── python/             # Estructura para la capa con dependencias
├── notebooks/              # Jupyter Notebooks de desarrollo progresivo
│   ├── nivel1_text_agent.ipynb
│   └── nivel2_voice_agent.ipynb
├── src/                    # Código fuente del backend
│   └── api/
│       ├── lambda_function.py # Lógica principal del agente para AWS Lambda
│       ├── products.json      # Copia del catálogo para empaquetar con la Lambda
│       └── requirements.txt   # Dependencias Python para la capa de Lambda
├── .env.example            # Ejemplo de variables de entorno
├── .gitignore
└── README.md               # Este archivo


## Tecnologías Utilizadas

* **Frontend:** HTML, JavaScript (API `MediaRecorder`), Tailwind CSS (CDN).
* **Backend:**
    * AWS Lambda (Python 3.11)
    * Amazon API Gateway (HTTP API)
    * Amazon Transcribe (Voz a Texto)
    * Amazon Bedrock (Modelo Titan Text Express para NLU y Generación de Respuesta)
    * Amazon Polly (Texto a Voz)
    * Amazon S3 (Almacenamiento temporal para Transcribe y para la capa de Lambda)
* **Orquestación del Agente:** LangGraph (framework Python sobre LangChain).
* **Gestión de Dependencias Python:** `pip`, `requirements.txt`.
* **Control de Versiones:** Git, GitHub.

## Configuración y Despliegue

### Prerrequisitos

* Cuenta de AWS con permisos para Lambda, API Gateway, S3, Transcribe, Bedrock, Polly, e IAM.
* AWS CLI configurada localmente (opcional, pero útil).
* Python 3.11 (o la versión correspondiente al runtime de Lambda) instalado localmente.
* `pip` (manejador de paquetes de Python).
* Un bucket S3 en la misma región que los servicios de AWS que se utilizarán.

### 1. Configuración Local (Para Notebooks y Entorno Lambda)

1.  **Clonar el Repositorio:**
    ```bash
    git clone [URL_DE_TU_REPO_GITHUB]
    cd asistente-compras-inteligente
    ```
2.  **Crear y Activar Entorno Virtual:**
    ```bash
    python -m venv venv
    # En Windows:
    venv\Scripts\activate
    # En macOS/Linux:
    source venv/bin/activate
    ```
3.  **Instalar Dependencias (para los notebooks):**
    ```bash
    pip install -r notebooks/requirements.txt 
    # (Asegúrate de tener un requirements.txt para los notebooks o instala manualmente:
    # pip install jupyter notebook boto3 python-dotenv langchain-aws langgraph requests)
    ```
4.  **Configurar Credenciales de AWS:**
    * Crea un archivo `.env` en la raíz del proyecto a partir de `.env.example`.
    * Rellena tus `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, y `AWS_DEFAULT_REGION`.
        ```
        AWS_ACCESS_KEY_ID="TU_ACCESS_KEY"
        AWS_SECRET_ACCESS_KEY="TU_SECRET_KEY"
        AWS_DEFAULT_REGION="us-east-1" # O tu región
        ```
    * Asegúrate de tener acceso a los modelos de Bedrock (ej. Titan Text Express) y a las voces de Polly (ej. Lupe) en la región especificada.

### 2. Despliegue del Backend (AWS Lambda y API Gateway)

#### a. Preparar la Capa de Lambda para Dependencias

Las dependencias de Python (LangChain, LangGraph, etc.) se empaquetarán en una capa de Lambda.

1.  **Crear Estructura de Carpetas para la Capa:**
    En la raíz de tu proyecto, crea:
    `lambda_layer_stuff/python/lib/python3.11/site-packages/`
    (Ajusta `python3.11` a tu runtime de Lambda si es diferente).

2.  **Instalar Dependencias en la Carpeta de la Capa:**
    Desde la raíz de tu proyecto (con `venv` activado):
    ```bash
    pip install -r src/api/requirements.txt --platform manylinux2014_x86_64 --python-version 3.11 --implementation cp --abi cp311 --only-binary=:all: -t ./lambda_layer_stuff/python/lib/python3.11/site-packages/
    ```
    (Ajusta `--python-version` y `--abi` si tu runtime Lambda es diferente a Python 3.11).

3.  **Crear el ZIP de la Capa:**
    Navega a la carpeta `lambda_layer_stuff/` y comprime la carpeta `python` en un archivo ZIP (ej. `langchain_lambda_layer.zip`). El ZIP debe contener la carpeta `python` en su raíz.

4.  **Subir el ZIP de la Capa a S3:**
    Sube `langchain_lambda_layer.zip` a un bucket S3 en la misma región que tu Lambda.

5.  **Crear la Capa de Lambda en AWS:**
    * En la consola de Lambda, ve a "Capas" y crea una nueva capa.
    * Proporciona un nombre (ej. `LangChainDependenciesLayer`).
    * Selecciona "Cargar un archivo desde Amazon S3" y proporciona la URI S3 de tu ZIP.
    * Elige los runtimes compatibles (ej. Python 3.11) y la arquitectura (ej. x86_64).
    * Crea la capa.

#### b. Preparar el Paquete de Despliegue de la Función Lambda

1.  **Asegurar Archivos en `src/api/`:**
    * `lambda_function.py` (código principal).
    * `products.json` (catálogo de productos).
    * (El `requirements.txt` en `src/api/` fue usado para la capa).

2.  **Crear el ZIP de la Función:**
    Navega a la carpeta `src/api/`. Selecciona `lambda_function.py` y `products.json` y comprímelos en un archivo ZIP (ej. `asistente_compras_lambda_package.zip`). Estos archivos deben estar en la raíz del ZIP.

#### c. Crear y Configurar la Función Lambda

1.  **Crear Función Lambda en AWS:**
    * Nombre: ej. `AsistenteComprasVoz`
    * Runtime: Python 3.11 (o el que corresponda)
    * Arquitectura: x86_64
    * Permisos: Crea un nuevo rol con permisos básicos de Lambda.
2.  **Subir Código:**
    * En "Código fuente", carga el `asistente_compras_lambda_package.zip`.
3.  **Configuración del Controlador:**
    * Handler: `lambda_function.lambda_handler`
4.  **Añadir Capa:**
    * En la sección "Capas", añade la capa que creaste (ej. `LangChainDependenciesLayer`).
5.  **Variables de Entorno:**
    * `S3_BUCKET_NAME`: Nombre de tu bucket S3 (ej. `chatvoice01`)
    * `BEDROCK_MODEL_ID`: `amazon.titan-text-express-v1`
    * `POLLY_VOICE_ID`: `Lupe`
    * (No necesitas `AWS_REGION` aquí, Lambda la toma del entorno).
6.  **Configuración General:**
    * Memoria: 512 MB o 1024 MB.
    * Tiempo de espera: 3 a 5 minutos (para permitir Transcribe y otras operaciones).
7.  **Permisos del Rol IAM de Lambda:**
    * Edita el rol IAM creado para la Lambda.
    * Adjunta políticas para permitir acceso a: S3 (GetObject, PutObject, DeleteObject para el bucket especificado), Transcribe (StartTranscriptionJob, GetTranscriptionJob), Bedrock (InvokeModel), Polly (SynthesizeSpeech). La política `AWSLambdaBasicExecutionRole` ya debería estar para los logs.

#### d. Crear y Configurar API Gateway (HTTP API)

1.  **Crear HTTP API en AWS API Gateway:**
    * Nombre: ej. `AsistenteVozAPI-v3`
2.  **Integración:**
    * Crea una integración de tipo Lambda apuntando a tu función `AsistenteComprasVoz`.
3.  **Rutas:**
    * Crea una ruta `POST` con un path como `/interactuar`.
    * Asocia esta ruta a la integración Lambda.
4.  **Etapas:**
    * Usa la etapa `$default` con despliegue automático.
5.  **CORS:**
    * Configura CORS para permitir solicitudes desde `*` (para pruebas) o tu dominio de frontend.
    * Métodos permitidos: `POST, OPTIONS`.
    * Cabeceras permitidas: `Content-Type` (y otras si son necesarias).
6.  **Obtener URL de Invocación:**
    * Copia la URL de invocación de tu API (ej. `https://api-id.execute-api.region.amazonaws.com/interactuar`).

### 3. Configurar y Ejecutar la Interfaz de Usuario (`frontend/voice_ui.html`)

1.  Edita `frontend/voice_ui.html`.
2.  Actualiza la constante `API_GATEWAY_INVOKE_URL` con la URL completa de tu endpoint de API Gateway.
3.  Abre `voice_ui.html` en un navegador web.
4.  Haz clic en "Iniciar Grabación", habla tu consulta, y luego "Detener Grabación".
5.  Observa la respuesta textual y escucha la respuesta en audio.

## Flujo de la Aplicación

1.  Usuario graba audio en `voice_ui.html`.
2.  La UI envía el audio (base64) a API Gateway.
3.  API Gateway invoca la función Lambda.
4.  Lambda:
    a.  Decodifica el audio y lo sube a S3.
    b.  Inicia un trabajo en Amazon Transcribe.
    c.  Espera y obtiene el texto transcrito.
    d.  Pasa el texto al agente LangGraph.
    e.  Agente LangGraph (NLU -> Catálogo -> Generación de Respuesta con Bedrock).
    f.  Toma la respuesta textual y la envía a Amazon Polly.
    g.  Obtiene el audio de Polly.
    h.  Devuelve el texto y el audio (base64) a API Gateway.
5.  API Gateway devuelve la respuesta a la UI.
6.  La UI muestra el texto y reproduce el audio.

## Próximos Pasos y Mejoras Futuras (Mencionado en Documento Técnico)

* Implementación de conversación multi-turno.
* Streaming de audio en tiempo real con tecnologías como WebRTC (ej. **LiveKit**).
* Arquitectura asíncrona para procesos largos (Transcribe, Polly).
* Conexión a bases de datos reales y APIs externas.
* Interfaz de usuario más avanzada y con mejor feedback.
* Seguridad robusta (autenticación/autorización para API Gateway).


