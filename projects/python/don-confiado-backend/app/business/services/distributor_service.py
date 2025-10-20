import os

from langchain_google_genai import ChatGoogleGenerativeAI
from supabase import create_client, Client

from business.utils.llm_utils import LLMUtils
from endpoints.dto.message_dto import (ChatRequestDTO)

class DistributorService():

    def create(self, request: ChatRequestDTO, llm: ChatGoogleGenerativeAI) -> dict[str, any]:
        # Rama 'Create_distribuitor': validar completitud y luego extraer datos     
        # 1) Verificación de completitud
        completeness_schema = {
            "title": "DistribuidorCompleteness",
            "type": "object",
            "properties": {
                "is_complete": {"type": "boolean"},
                "missing_fields": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "tipo_documento",
                            "numero_documento",
                            "razon_social",
                            "nombres",
                            "apellidos"
                        ]
                    }
                }
            },
            "required": ["is_complete", "missing_fields"],
            "additionalProperties": False,
        }       

        completeness_model = llm.with_structured_output(completeness_schema)
        completeness_text = (
            "Evalúa si el mensaje contiene la información completa para crear un distribuidor. "
            "Requisitos: tipo_documento (CC/NIT/CE), numero_documento y (razon_social) o (nombres y apellidos). "
            "Devuelve is_complete=true solo si todos los requisitos están presentes en el mensaje. "
            "Si falta algo, lista los campos faltantes en missing_fields.") + f"\n\nMensaje del usuario: {request.message}"     
      
        completeness = completeness_model.invoke(completeness_text)
        print(completeness)    

        is_complete = bool(completeness[0]["args"].get("is_complete", False))
        missing_fields = completeness[0]["args"].get("missing_fields", []) or []

        if not is_complete:
            # Solicitud de datos faltantes (prompt plano + memoria)
            history_text = LLMUtils._history_as_text(request.user_id)
            user_input = request.message        
            request_missing_text = (
                "ROLE: Don Confiado, asesor empresarial amable y claro.\n"
                "Pide al usuario, en una sola oración y sin tecnicismos, los datos faltantes: "
                f"{', '.join(missing_fields)}.\n\n"
                f"Historial:\n{history_text}\n\n"
                f"Usuario: {user_input}\n"
                f"Asistente:"
            )       

            reply_obj = llm.invoke(request_missing_text)
            reply_text = getattr(reply_obj, "content", str(reply_obj))

            LLMUtils._append_message(request.user_id, "ai", reply_text)     
            
            return {
                "userintention": "Create_distribuitor",
                "status": "need_more_data",
                "missing_fields": missing_fields,
                "reply": reply_text,
            }
            
        # 2) Extracción de datos (solo cuando está completo)
        extraction_schema = {
            "title": "DistribuidorData",
            "description": (
                "Extra unicamente los campos que el usuario proporciona. No inventes valores."
            ),
            "type": "object",
            "properties": {
                "tipo_documento": {
                    "type": "string",
                    "enum": ["CC", "NIT", "CE"],
                    "description": "Tipo de documento: CC, NIT o CE"
                },
                "numero_documento": {"type": "string"},
                "razon_social": {"type": "string"},
                "nombres": {"type": "string"},
                "apellidos": {"type": "string"},
                "telefono_fijo": {"type": "string"},
                "telefono_celular": {"type": "string"},
                "direccion": {"type": "string"},
                "email": {"type": "string"}
            },
            "additionalProperties": False,
        }

        extractor = llm.with_structured_output(extraction_schema)
        extract_text = (
            "Extrae los campos del distribuidor desde el mensaje del usuario. No inventes datos. "
            "Si un campo no está presente, omítelo (no devuelvas null).\n\n"
            f"Mensaje del usuario: {request.message}"
        )

        extracted_payload = extractor.invoke(extract_text)
        print(extracted_payload)

        extracted = extracted_payload[0]["args"] if isinstance(extracted_payload, list) else extracted_payload
        tipo_documento = extracted.get("tipo_documento")
        numero_documento = extracted.get("numero_documento")
        razon_social = extracted.get("razon_social")
        nombres = extracted.get("nombres")
        apellidos = extracted.get("apellidos")      
        
        # Sanitizar registro (evitar null, vacíos y "null")
        def _valid_value(value: object) -> bool:
            if value is None: return False      
            text = str(value).strip()       
            
            if text == "": return False     
            if text.lower() == "null": return False     
            
            return True     
        
        # Validación credenciales Supabase
        supabase_url = os.getenv("SUPABASE_URL")
        print(f"supabase_url={supabase_url}")       
        
        upabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        print(f"supabase_key={supabase_key}")       
        
        if not supabase_url or not supabase_key:
            # Respuesta breve informando falta de credenciales
            user_input = request.message        
            creds_text = (
                "ROLE: Don Confiado, asesor empresarial.\n"
                "Informa brevemente que faltan las credenciales de Supabase (SUPABASE_URL / "
                "SUPABASE_SERVICE_ROLE_KEY) y que deben configurarse antes de continuar.\n\n"
                f"Usuario: {user_input}\n"
                f"Asistente:"
            )       
            reply_obj = llm.invoke(creds_text)
            reply_text = getattr(reply_obj, "content", str(reply_obj))
            
            LLMUtils._append_message(request.user_id, "ai", reply_text)     
            
            return {
                "userintention": "Create_distribuitor",
                "status": "error",
                "error": "Missing Supabase credentials",
                "reply": reply_text,
                "extracted": extracted,
            }       
        # Inicialización cliente Supabase
        global _supabase_client

        try:
            _supabase_client
        except NameError:
            _supabase_client = create_client(supabase_url, supabase_key)        
        record = {k: v for k, v in extracted.items() if _valid_value(v)}
        record["tipo_tercero"] = "proveedor"        
        
        try:
            # Inserción en Supabase y confirmación
            response = _supabase_client.table("terceros").insert(record).execute()
            data = getattr(response, "data", None)      
            user_input = request.message        
            confirm_text = (
                "ROLE: Don Confiado, asesor empresarial.\n"
                "Confirma brevemente que el distribuidor ha sido creado exitosamente.\n\n"
                f"Usuario: {user_input}\n"
                f"Asistente:"
            )
            reply_obj = llm.invoke(confirm_text)
            reply_text = getattr(reply_obj, "content", str(reply_obj))

            LLMUtils._append_message(request.user_id, "ai", reply_text)

            return {
                "userintention": "Create_distribuitor",
                "status": "created",
                "data": data,
                "reply": reply_text,
            }
        except Exception as ex:
            # Manejo de error al crear distribuidor (prompt plano)
            user_input = request.message        
            error_text = (
                "ROLE: Don Confiado, asesor empresarial empático.\n"
                "Informa que ocurrió un error al crear el distribuidor y que intente de nuevo, "
                "sin detalles técnicos.\n\n"
                f"Usuario: {user_input}\n"
                f"Asistente:"
            )
            reply_obj = llm.invoke(error_text)
            reply_text = getattr(reply_obj, "content", str(reply_obj))
            
            LLMUtils._append_message(request.user_id, "ai", reply_text)     
            
            return {
                "userintention": "Create_distribuitor",
                "status": "error",
                "error": str(ex),
                "reply": reply_text,
                "extracted": extracted,
            }