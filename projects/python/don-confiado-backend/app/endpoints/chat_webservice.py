from fastapi import APIRouter, HTTPException
from fastapi_utils.cbv import cbv

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
"""Chat endpoints sin utilizar helpers de memoria de LangChain.

Se usa un almacenamiento en memoria simple (dict + listas) por usuario
para construir el contexto de conversación y se generan prompts como cadenas.
"""

from endpoints.dto.message_dto import (ChatRequestDTO)
from supabase import create_client, Client

from business.utils.llm_utils import LLMUtils
from business.services.basic_service import BasicService
from business.services.product_service import ProductService
from business.services.distributor_service import DistributorService

# --- Configuración de entorno ---
load_dotenv()

# --- Router y clase del servicio de chat ---
chat_webservice_api_router = APIRouter()

@cbv(chat_webservice_api_router)
class ChatWebService:
    # --- v1.0: Chat con memoria en sesión ---
    @chat_webservice_api_router.post("/api/chat_v1.0")
    async def chat_with_memory(self, request: ChatRequestDTO):
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            api_key = input("Por favor, ingrese su API KEY de Google (GOOGLE_API_KEY): ")
            os.environ["GOOGLE_API_KEY"] = api_key

        # Modelo y prompt del sistema
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

        return BasicService().process(request, llm);


    # --- v1.1: Clasificación de intención + extracción y registro de distribuidor ---
    @chat_webservice_api_router.post("/api/chat_v1.1")
    async def chat_with_structure_output(self, request: ChatRequestDTO):
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            api_key = input("Por favor, ingrese su API KEY de Google (GOOGLE_API_KEY): ")
            os.environ["GOOGLE_API_KEY"] = api_key

        # Modelo base
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

        # Registrar el mensaje actual en memoria y construir historial
        user_input = request.message
        LLMUtils._append_message(request.user_id, "human", user_input)
        history_text = LLMUtils._history_as_text(request.user_id)

        # Esquema de intención + clasificador estructurado
        intention_schema = {
            "title": "UserIntention",
            "description": (
                "Clasifica la intención del mensaje del usuario. "
                "Devuelve solo una de las etiquetas permitidas."
            ),
            "type": "object",
            "properties": {
                "userintention": {
                    "type": "string",
                    "enum": ["Create_distribuitor", "Create_product", "Other"],
                    "description": (
                        "'Create_distribuitor': cuando el usuario quiere crear/registrar un proveedor/distribuidor."
                        "'Create_product': cuando el usuario quiere registrar un nuevo producto."
                        "'Other': conversación casual u otro propósito."
                    ),
                }
            },
            "required": ["userintention"],
            "additionalProperties": False,
        }

        model_with_structure = llm.with_structured_output(intention_schema)

        # Clasificación de intención (prompt plano)
        classify_text = (
            "Eres un clasificador. Lee la conversación y clasifica la intención "
            "estrictamente en una de tres etiquetas: 'Create_distribuitor', 'Create_product', u 'Other'. "
            "Usa 'Create_distribuitor' cuando el usuario pretende crear/registrar un proveedor/"
            "distribuidor (p. ej., menciona crear un proveedor/distribuidor)."
            "Usa 'Create_product' cuando el usuario pretende registrar un producto (p.e., menciona crear un producto)"
            "En otro caso usa 'Other'.\n\n"
            f"Historial:\n{history_text}\n\n"
            f"Último mensaje del usuario: {user_input}"
        )

        result = model_with_structure.invoke(classify_text)
        print(result)

        user_intention = result[0]["args"].get("userintention")

        if (user_intention == "Create_distribuitor"):
            return DistributorService().create(request, llm)        
        elif (user_intention == "Create_product"):
            return ProductService().create(request, llm)
        else:
            # Rama 'Other': respuesta general con memoria
            return BasicService().process(request=request, llm=llm, append_human_message=False, user_intention="Other")
