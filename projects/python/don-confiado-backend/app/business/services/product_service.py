import os

from langchain_google_genai import ChatGoogleGenerativeAI
from supabase import create_client, Client

from business.utils.llm_utils import LLMUtils
from endpoints.dto.message_dto import (ChatRequestDTO)

class ProductService():

    def _get_invoke_value(self, request: ChatRequestDTO, status: str, missing_fields: str = None):
        role_detail: str = ""

        if (status == "need_more_data"):
            history_text = LLMUtils._history_as_text(request.user_id)
            missing_fields_list = ', '.join(missing_fields)

            role_detail = f"Pide al usuario, en una sola oración y sin tecnicismos, los datos faltantes: {missing_fields_list}.\n\n Historial:\n{history_text}"       
        elif (status == "error1"):
            role_detail = "Informa brevemente que faltan las credenciales de Supabase (SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY) y que deben configurarse antes de continuar"       
        elif (status == "error2"):
            role_detail = "Informa que ocurrió un error al crear el producto y que intente de nuevo, sin detalles técnicos."
        elif (status == "created"):
            role_detail = "Confirma brevemente que el producto ha sido creado exitosamente."
        
        user_input: str = request.message

        return f"""
        ROLE: Don Confiado, asesor empresarial empático, amable y claro.\n
        {role_detail}\n\n
        Usuario: {user_input}\n
        Asistente:
        """
            
    def _build_response(self,
                        request: ChatRequestDTO,
                        llm: ChatGoogleGenerativeAI,
                        status: str,
                        missing_fields: str = None,
                        error: str = None,
                        extracted: str = None,
                        data: any = None):
        invoke_value: str = self._get_invoke_value(request=request, status=status, missing_fields=missing_fields)

        reply_obj = llm.invoke(invoke_value)
        reply_text = getattr(reply_obj, "content", str(reply_obj))

        LLMUtils._append_message(request.user_id, "ai", reply_text)
        
        resp: dict = {
            "userintention": "Create_product",
            "status": status,
            "reply": reply_text
        }

        if (missing_fields is not None): resp["missing_fields"] = missing_fields
        if (error is not None): resp["error"] = error
        if (extracted is not None): resp["extracted"] = extracted
        if (data is not None): resp["data"] = data

        return resp

    def create(self, request: ChatRequestDTO, llm: ChatGoogleGenerativeAI) -> dict[str, any]:
       # Rama 'Create_product': Validar completitud y luego extraer datos.

       # 1) Verificación de completitud
       completeness_schema = {
           "title": "ProductCompleteness",
           "type": "object",
           "properties": {
               "is_complete": {"type": "boolean"},
               "missing_fields": {
                   "type": "array",
                   "items": {
                       "type": "string",
                       "enum": [
                           "sku",
                           "nombre",
                           "precio_venta",
                           "cantidad",
                           "proveedor_id"
                       ]
                   }
               }
           },
           "required": ["is_complete", "missing_fields"],
           "additionalProperties": False,
       }

       completeness_model = llm.with_structured_output(completeness_schema)
       completeness_text = (
           "Evalúa si el mensaje contiene la información completa para crear un producto. "
           "Requisitos: SKU, nombre, precio de venta, cantidad y proveedor (Nombre y/o documento del proveedor). "
           "Devuelve is_complete=true solo si todos los requisitos están presentes en el mensaje. "
           "Si falta algo, lista los campos faltantes en missing_fields.") + f"\n\nMensaje del usuario: {request.message}"
       
       completeness = completeness_model.invoke(completeness_text)
       print(completeness)
       
       is_complete = bool(completeness[0]["args"].get("is_complete", False))
       missing_fields = completeness[0]["args"].get("missing_fields", []) or []
       
       if not is_complete:
           # Solicitud de datos faltantes (prompt plano + memoria)
           return self._build_response(request=request, llm=llm, status="need_more_data", missing_fields=missing_fields)

       # 2) Extracción de datos (solo cuando está completo)
       extraction_schema = {
           "title": "ProductData",
           "description": (
               "Extra unicamente los campos que el usuario proporciona. No inventes valores."
           ),
           "type": "object",
           "properties": {
               "sku": {"type": "string"},
               "nombre": {"type": "string"},
               "precio_venta": {"type": "number"},
               "cantidad": {"type": "integer"},
               "proveedor_id": {"type": "integer"}
           },
           "additionalProperties": False,
       }

       extractor = llm.with_structured_output(extraction_schema)
       extract_text = (
           "Extrae los campos del producto desde el mensaje del usuario. No inventes datos. "
           "Si un campo no está presente, omítelo (no devuelvas null).\n\n"
           f"Mensaje del usuario: {request.message}"
       )

       extracted_payload = extractor.invoke(extract_text)
       print(extracted_payload)
       
       extracted = extracted_payload[0]["args"] if isinstance(extracted_payload, list) else extracted_payload
       
       sku = extracted.get("sku")
       nombre = extracted.get("nombre")
       precio_venta = extracted.get("precio_venta")
       cantidad: int = int(extracted.get("cantidad"))
       proveedor_id = extracted.get("proveedor_id")
       
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
       
       supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
       print(f"supabase_key={supabase_key}")
       
       if ((not supabase_url) or (not supabase_key)):
           return self._build_response(request=request, llm=llm, status="error1", error="Missing Supabase credentials", extracted=extracted)

       # Inicialización cliente Supabase
       global _supabase_client

       try:
           _supabase_client
       except NameError:
           _supabase_client = create_client(supabase_url, supabase_key)

       record = {k: v for k, v in extracted.items() if _valid_value(v)}
       record["cantidad"] = int(record["cantidad"])
       record["proveedor_id"] = int(record["proveedor_id"])

       try:
           # Inserción en Supabase y confirmación
           response = _supabase_client.table("productos").insert(record).execute()
           data = getattr(response, "data", None)
                    
           return self._build_response(request=request,llm=llm,status="created", data=data)
       except Exception as ex:
           # Manejo de error al crear distribuidor (prompt plano)
           print(ex)
                   
           return self._build_response(request=request, llm=llm, status="error2", error=str(ex), extracted=extracted)